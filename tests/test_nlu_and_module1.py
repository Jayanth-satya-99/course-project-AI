import unittest

from module1.csp import CSPSolver
from module1.evidence_graph import EvidenceGraph
from module1.minimax import MinimaxEngine
from module1.policy_graph import PolicyGraph
from module4.nlu import NLUExtractor
from orchestrator import PolicyAdherentOrchestrator
from shared.models import ActionType, ConstraintStatus, ThreatLevel


class NLUAndModule1ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.policy_graph = PolicyGraph()
        self.evidence_graph = EvidenceGraph(self.policy_graph)
        self.csp = CSPSolver(self.policy_graph)
        self.nlu = NLUExtractor()
        self.minimax = MinimaxEngine()

    def test_nlu_extracts_advanced_components(self):
        text = (
            "Please help me, I am a single mother and my child needs medicine. "
            "Refund my $500 TV immediately or I will sue your company and contact my lawyer! "
            "I have the purchase receipt."
        )
        history = [{"role": "assistant", "content": "How can I help you today?"}]
        claim = self.nlu.extract(text, conversation_history=history)

        self.assertEqual(claim.intent, "refund")
        self.assertEqual(claim.amount, 500.0)
        self.assertIn("receipt", claim.evidence_provided)

        # Check emotional appeals
        self.assertTrue(len(claim.emotional_appeals) > 0)
        self.assertIn("single_parent_family", claim.emotional_appeals)
        self.assertIn("medical_hardship", claim.emotional_appeals)

        # Check threats
        self.assertTrue(len(claim.threats) > 0)
        self.assertIn("legal_action", claim.threats)

        # Check persuasive strategies
        self.assertTrue(len(claim.persuasive_strategies) > 0)
        self.assertIn("urgency_pressure", claim.persuasive_strategies)
        self.assertIn("sympathy_exploitation", claim.persuasive_strategies)
        self.assertIn("threat_intimidation", claim.persuasive_strategies)

        # Check supporting evidence
        self.assertEqual(len(claim.supporting_evidence), 1)
        self.assertEqual(claim.supporting_evidence[0]["evidence_type"], "receipt")
        self.assertIn("receipt", claim.supporting_evidence[0]["snippet"])

        # Check conversation history preservation
        self.assertEqual(len(claim.conversation_history), 1)
        self.assertEqual(claim.conversation_history[0]["content"], "How can I help you today?")

    def test_contradictory_claims_intra_statement(self):
        contradictory_text = (
            "The package was factory sealed and completely unopened, "
            "but when I checked the screen inside was cracked and shattered."
        )
        claim = self.nlu.extract(contradictory_text)
        self.assertTrue(len(claim.contradictory_claims) > 0)
        self.assertEqual(claim.contradictory_claims[0]["type"], "claim_statement_conflict")
        self.assertIn("contradiction", claim.adversarial_tactics)
        self.assertIn("fact_shifting", claim.persuasive_strategies)

    def test_contradictory_claims_cross_turn(self):
        history = [
            {"role": "user", "content": "I paid $500 for order ORD-001 last week."},
        ]
        turn2_text = "I need a refund of $1200 for order ORD-002 right now."
        claim = self.nlu.extract(turn2_text, conversation_history=history)

        conflict_types = [c["type"] for c in claim.contradictory_claims]
        self.assertIn("order_mismatch", conflict_types)
        self.assertIn("contradiction", claim.adversarial_tactics)

    def test_module1_csp_manipulation_flag_from_contradiction(self):
        # A claim with an order mismatch across turns
        history = [
            {"role": "user", "content": "I paid $500 for order ORD-001 last week."},
        ]
        turn2_text = "I need a refund for order ORD-002 right now."
        claim = self.nlu.extract(turn2_text, conversation_history=history)

        csp_res = self.csp.evaluate(claim)
        self.assertTrue(csp_res.has_manipulated)
        self.assertTrue(any(
            status == ConstraintStatus.POTENTIALLY_MANIPULATED
            for status in csp_res.constraints.values()
        ))

    def test_module1_evidence_graph_integration(self):
        text = "I bought a TV for $500 under ORD-001. I have the receipt and photo."
        claim = self.nlu.extract(text)
        self.assertEqual(len(claim.supporting_evidence), 2)

        evidence_res = self.evidence_graph.search(claim)
        # Check that provided nodes in evidence graph include snippet & confidence
        provided_nodes = evidence_res.provided_evidence
        receipt_node = next((n for n in provided_nodes if n["type"] == "receipt"), None)
        self.assertIsNotNone(receipt_node)
        self.assertIn("snippet", receipt_node)
        self.assertGreaterEqual(receipt_node["confidence"], 0.90)

    def test_module1_minimax_reacts_to_nlu_threats_and_appeals(self):
        text = "Refund my $500 immediately or I will sue your company and contact my lawyer!"
        claim = self.nlu.extract(text)
        self.assertIn("legal_action", claim.threats)

        res = self.minimax.search(ActionType.ASK_FOR_EVIDENCE, claim, ["REF_C1"])
        self.assertIsNotNone(res.selected_response_type)
        self.assertGreater(res.evaluated_nodes, 0)


if __name__ == "__main__":
    unittest.main()

