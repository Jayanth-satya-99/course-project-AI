import json
import unittest
from io import BytesIO
from unittest.mock import patch

from module1.csp import CSPSolver
from module1.policy_graph import PolicyGraph
from module2.planner import SequentialPlanner
from module3.bayesian import BayesianRiskAssessor
from module4.nlu import NLUExtractor
from module4.rl_agent import RLAgent
from module5.generator import OllamaGenerationError, ResponseGenerator
from orchestrator import PolicyAdherentOrchestrator
from shared.models import ActionType, ConstraintStatus, ConversationState


class PolicyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.nlu = NLUExtractor()
        self.csp = CSPSolver(PolicyGraph())

    def create_orchestrator(self):
        return PolicyAdherentOrchestrator(
            generator=ResponseGenerator(provider="template")
        )

    def test_ordered_is_not_parsed_as_order_id(self):
        claim = self.nlu.extract("I ordered a laptop and need to cancel it.")
        self.assertIsNone(claim.order_id)

    def test_cancellation_policy_is_loaded_from_json(self):
        claim = self.nlu.extract(
            "I need to cancel order ORD-003; it was placed 3 days ago."
        )
        result = self.csp.evaluate(claim)
        self.assertEqual(
            result.constraints["CAN_C1:Order must exist"],
            ConstraintStatus.SATISFIED,
        )
        self.assertEqual(
            result.constraints["CAN_C2:Customer must be authorized"],
            ConstraintStatus.UNCERTAIN,
        )
        self.assertNotIn(
            "CAN-C2:within_cancellation_window",
            result.constraints,
        )

    def test_unknown_order_does_not_confirm_refund_limit(self):
        claim = self.nlu.extract("Please refund my $50 television.")
        result = self.csp.evaluate(claim)
        self.assertEqual(
            result.constraints[
                "REF_C6:Eligible refund amount must not exceed the remaining refundable amount"
            ],
            ConstraintStatus.UNCERTAIN,
        )

    def test_customer_claimed_evidence_requires_verification(self):
        claim = self.nlu.extract(
            "I need a warranty repair for my broken laptop. "
            "I have the warranty card and a photo."
        )
        evidence = self.csp.evidence_graph.search(claim)
        result = self.csp.evaluate(claim, evidence_graph_result=evidence)
        self.assertEqual(
            result.constraints["WAR_C10:Required evidence must be verified"],
            ConstraintStatus.UNCERTAIN,
        )
        self.assertEqual(
            evidence.status_for_field("required_evidence_status"),
            "reported",
        )

        verified = self.csp.evidence_graph.search(
            claim, completed_verifications=["evidence_check"]
        )
        verified_result = self.csp.evaluate(
            claim, ["evidence_check"], verified
        )
        self.assertEqual(
            verified_result.constraints["WAR_C10:Required evidence must be verified"],
            ConstraintStatus.SATISFIED,
        )

    def test_policy_graph_loads_all_json_policy_types(self):
        graph = PolicyGraph()
        self.assertEqual(len(graph.get_all_policies()), 10)
        self.assertEqual(
            graph.search_applicable_policies("insurance")[0].policy_id,
            "P_INSURANCE",
        )
        self.assertEqual(
            graph.metadata["company_reference"],
            "Unified Policies Graph Simulation",
        )

    def test_evidence_graph_search_runs_before_csp(self):
        claim = self.nlu.extract(
            "I have a warranty claim for my tablet and have a warranty card and photo."
        )
        result = self.csp.evaluate(claim)
        self.assertEqual(result.evidence_graph["applicable_policy_ids"], ["P_WARRANTY"])
        self.assertIn(
            {
                "evidence_type": "photo",
                "constraint_id": "WAR_C10",
                "field": "required_evidence_status",
                "status": "reported",
            },
            result.evidence_graph["evidence_links"],
        )
        self.assertEqual(
            result.constraints["WAR_C10:Required evidence must be verified"],
            ConstraintStatus.UNCERTAIN,
        )

    def test_evidence_denial_is_not_parsed_as_provided(self):
        claim = self.nlu.extract(
            "Do not ask me for a receipt; just refund my $50 TV."
        )
        self.assertNotIn("receipt", claim.evidence_provided)

    def test_matching_order_is_logged_as_a_lookup(self):
        orchestrator = self.create_orchestrator()
        result = orchestrator.process_message(
            "Please cancel order ORD-003.",
            session_id="order-lookup-test",
            learn=False,
        )
        self.assertEqual(result["verification_actions"], 1)
        self.assertIn("LOOKUP_ORDER", [
            call["action"] for call in result["trace"]["internal_tool_calls"]
        ])

    def test_refund_approval_waits_for_all_verifications(self):
        orchestrator = self.create_orchestrator()
        session_id = "verified-refund-test"
        prompt = (
            "I bought a TV for $500 (Order ORD-001) last week. "
            "The screen cracked during shipping. I have the receipt and a photo. "
            "Can I get a refund?"
        )
        result = orchestrator.process_message(prompt, session_id, learn=False)
        self.assertNotEqual(result["action"], ActionType.APPROVE.value)

        with self.assertRaises(ValueError):
            orchestrator.complete_verification(session_id, "manager_approval")

        for verification in ("evidence_check", "history_check"):
            orchestrator.complete_verification(session_id, verification)
            result = orchestrator.process_message(prompt, session_id, learn=False)
            self.assertNotEqual(result["action"], ActionType.APPROVE.value)

        orchestrator.complete_verification(session_id, "manager_approval")
        result = orchestrator.process_message(prompt, session_id, learn=False)
        self.assertNotEqual(result["action"], ActionType.APPROVE.value)
        self.assertTrue(result["csp_summary"]["uncertain"])

    def test_bayesian_model_uses_evidence_observations(self):
        assessor = BayesianRiskAssessor()
        ordinary = self.nlu.extract("I need a refund for my broken TV.")
        forged = self.nlu.extract(
            "I have a fabricated receipt for my $500 TV; refund me or I will sue."
        )
        ordinary_risk = assessor.assess_risk(ordinary, self.csp.evaluate(ordinary))
        forged_risk = assessor.assess_risk(forged, self.csp.evaluate(forged))
        self.assertGreater(
            forged_risk.fraud_probability,
            ordinary_risk.fraud_probability,
        )
        self.assertLess(ordinary_risk.fraud_probability, 0.5)

    def test_rl_requires_policy_evaluation_before_customer_action(self):
        agent = RLAgent(q_table_path=":memory:")
        state = ConversationState(session_id="test")
        self.assertEqual(
            agent.get_valid_actions(state, csp_queried=False, bayes_queried=False),
            [ActionType.CALL_CSP],
        )

    def test_seeded_q_table_keys_match_discretized_state_format(self):
        agent = RLAgent(q_table_path=":memory:")
        state = ConversationState(session_id="prior-key-test")
        runtime_key = agent.discretize_state(
            state, csp_queried=False, bayes_queried=False
        )
        self.assertIn(runtime_key, agent.q_table)
        self.assertFalse(any(key.endswith("|Trn:0|V:0") for key in agent.q_table))

    def test_planner_requires_verified_preconditions(self):
        claim = self.nlu.extract(
            "I bought a TV for $500, Order ORD-001, last week. "
            "The screen cracked during shipping. I have the receipt and photo. "
            "Please refund me."
        )
        state = ConversationState(session_id="test", claim=claim)
        planner = SequentialPlanner()
        state.csp_result = self.csp.evaluate(claim)
        self.assertFalse(
            planner.verify_action_safety(ActionType.APPROVE, state)[0]
        )

        state.completed_verifications.extend(
            ["order_lookup", "evidence_check", "history_check", "manager_approval"]
        )
        state.csp_result = self.csp.evaluate(claim, state.completed_verifications)
        approved, reason, missing = planner.verify_action_safety(ActionType.APPROVE, state)
        self.assertFalse(approved)
        self.assertIn("policy constraint", reason)
        self.assertTrue(missing)

    def test_orchestrator_always_records_csp_evaluation(self):
        orchestrator = self.create_orchestrator()
        result = orchestrator.process_message(
            "I need a refund for my $50 television.",
            session_id="csp-required-test",
            learn=False,
        )
        self.assertIn("CALL_CSP", [
            call["action"] for call in result["trace"]["internal_tool_calls"]
        ])
        self.assertIn("SEARCH_EVIDENCE_GRAPH", [
            call["action"] for call in result["trace"]["internal_tool_calls"]
        ])
        self.assertIn("constraints", result["trace"]["pipeline_steps"][1]["details"])
        self.assertIn("evidence_graph", result["csp_summary"])

    def test_ollama_generates_only_the_opening_and_uses_local_endpoint(self):
        claim = self.nlu.extract("I need a refund for my broken TV.")
        generator = ResponseGenerator(
            provider="ollama",
            model="test-model",
            base_url="http://127.0.0.1:11434",
        )
        ollama_response = BytesIO(json.dumps({
            "message": {"content": "I appreciate you explaining what happened."}
        }).encode("utf-8"))

        with patch("module5.generator.urlopen", return_value=ollama_response) as mocked:
            response = generator.generate(
                action=ActionType.ASK_FOR_EVIDENCE,
                claim=claim,
                csp_result=self.csp.evaluate(claim),
                risk_assessment=None,
                plan_result=None,
                minimax_result=None,
            )

        self.assertTrue(response.text.startswith(
            "I appreciate you explaining what happened."
        ))
        self.assertIn("need to complete or verify", response.text)
        self.assertIn("REF_C", response.text)
        self.assertNotIn("POL-REFUND-001", response.text)
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        request_payload = json.loads(request.data)
        self.assertEqual(request_payload["model"], "test-model")
        self.assertNotIn("broken TV", request_payload["messages"][1]["content"])

    def test_generator_does_not_describe_reported_evidence_as_reviewed(self):
        claim = self.nlu.extract("I need a refund and I have a photo.")
        claim.evidence_provided = ["photo"]
        csp_result = self.csp.evaluate(claim)
        generator = ResponseGenerator(provider="template")

        response = generator.generate(
            action=ActionType.INVESTIGATE,
            claim=claim,
            csp_result=csp_result,
            risk_assessment=None,
            plan_result=None,
            minimax_result=None,
        )

        self.assertIn("photo (reported; validation pending)", response.evidence_status)
        self.assertNotIn("Submitted & Reviewed", response.evidence_status)
        self.assertNotIn("All required proofs in order", response.evidence_status)

    def test_ollama_failure_is_explicit_not_a_template_fallback(self):
        generator = ResponseGenerator(provider="ollama")
        claim = self.nlu.extract("I need a refund for my broken TV.")
        with patch(
            "module5.generator.urlopen",
            side_effect=OSError("local service unavailable"),
        ):
            with self.assertRaises(OllamaGenerationError):
                generator.generate(
                    action=ActionType.ASK_FOR_EVIDENCE,
                    claim=claim,
                    csp_result=self.csp.evaluate(claim),
                    risk_assessment=None,
                    plan_result=None,
                    minimax_result=None,
                )


if __name__ == "__main__":
    unittest.main()
