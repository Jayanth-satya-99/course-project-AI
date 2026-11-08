import json
import os
import tempfile
import unittest
from pathlib import Path

from module1.csp import CSPSolver
from module1.policy_graph import PolicyGraph
from module3.bayesian import BayesianRiskAssessor
from module4.nlu import NLUExtractor
from orchestrator import PolicyAdherentOrchestrator
from shared.models import (
    ClaimExtraction, CSPResult, ConstraintStatus, ConversationState,
    EvidenceStrength, ThreatLevel, Sentiment, RiskAssessment
)


class Module3BayesianNetworkTests(unittest.TestCase):
    def setUp(self):
        self.assessor = BayesianRiskAssessor()
        self.nlu = NLUExtractor()
        self.csp = CSPSolver(PolicyGraph())
        self.orchestrator = PolicyAdherentOrchestrator()

    def test_assertion_1_missing_evidence_baseline(self):
        """Unprovided evidence triggers EvidenceStrength.MISSING without hardcoding."""
        claim = self.nlu.extract("I want a refund for my item.")
        self.assertEqual(claim.evidence_provided, [])

        csp_result = self.csp.evaluate(claim)
        risk = self.assessor.assess_risk(claim, csp_result)

        self.assertEqual(risk.evidence_strength, EvidenceStrength.MISSING)
        self.assertEqual(risk.evidence_reliability, 0.0)
        self.assertIn("No supporting evidence provided", risk.risk_factors)

    def test_assertion_2_temporal_mismatch(self):
        """Violating return window in CSP forces temporal_consistency='contradictory', driving P(fraud) > 0.50."""
        # Dynamic pipeline: customer claims return for order ORD-004 delivered 45 days ago (window 30 days)
        claim = self.nlu.extract(
            "I want to return an item from order ORD-004 that was delivered 45 days ago. I have the receipt."
        )
        csp_result = self.csp.evaluate(claim)
        self.assertTrue(len(csp_result.violated_policies) > 0)

        risk = self.assessor.assess_risk(claim, csp_result)

        self.assertGreater(risk.posterior_details["P(fraudulent)"] + risk.posterior_details["P(suspicious)"], 0.50)
        self.assertGreater(risk.fraud_probability, 0.50)
        self.assertIn("P(temporal_aligned)", risk.posterior_details)
        self.assertEqual(risk.posterior_details["P(temporal_aligned)"], 0.0)
        self.assertTrue(any("temporal" in rf.lower() for rf in risk.risk_factors))

    def test_assertion_3_statement_contradiction(self):
        """NLU flagging 'contradiction' sets statement_match='contradictory'."""
        claim = self.nlu.extract(
            "The package is unopened and factory sealed, but the screen inside is completely cracked."
        )
        self.assertIn("contradiction", claim.adversarial_tactics)

        csp_result = self.csp.evaluate(claim)
        risk = self.assessor.assess_risk(claim, csp_result)

        self.assertEqual(risk.posterior_details["P(statement_consistent)"], 0.0)
        self.assertEqual(risk.evidence_strength, EvidenceStrength.CONTRADICTORY)
        self.assertTrue(any("contradiction" in rf.lower() for rf in risk.risk_factors))

    def test_assertion_4_deliberate_fabrication(self):
        """POTENTIALLY_MANIPULATED constraint status forces evidence_quality='fabricated', yielding P(fraud) >= 0.85."""
        # Dynamic pipeline: customer claims refund with fabricated receipt and threat
        claim = self.nlu.extract(
            "I have a fabricated receipt for my $500 TV under order ORD-001; refund me now or I will sue!"
        )
        self.assertIn("fabricated_evidence", claim.adversarial_tactics)

        csp_result = self.csp.evaluate(claim)
        self.assertTrue(csp_result.has_manipulated)

        risk = self.assessor.assess_risk(claim, csp_result)

        self.assertGreaterEqual(risk.fraud_probability, 0.85)
        self.assertEqual(risk.evidence_strength, EvidenceStrength.FABRICATED)
        self.assertGreater(risk.posterior_details["P(evidence_fabricated)"], 0.90)
        self.assertTrue(risk.recommendation.startswith("HIGH RISK"))

    def test_assertion_5_state_vector_compatibility(self):
        """Verify state.to_state_vector() extracts fraud_prob and evidence_reliability without schema mismatch."""
        claim = self.nlu.extract("I need a refund for my order ORD-001. I have the receipt.")
        csp_result = self.csp.evaluate(claim)
        risk = self.assessor.assess_risk(claim, csp_result)

        state = ConversationState(
            session_id="test_session_vec",
            turn=1,
            claim=claim,
            csp_result=csp_result,
            risk_assessment=risk,
        )

        state_vector = state.to_state_vector()

        self.assertIsInstance(state_vector, dict)
        self.assertIn("fraud_prob", state_vector)
        self.assertIn("evidence_reliability", state_vector)
        self.assertEqual(state_vector["fraud_prob"], risk.fraud_probability)
        self.assertEqual(state_vector["evidence_reliability"], risk.evidence_reliability)
        for val in state_vector.values():
            self.assertIsInstance(val, (float, int))

    def test_completed_verification_clamps_evidence_strong(self):
        """If 'evidence_check' is present in completed_verifications, evidence_quality is clamped to strong."""
        claim = self.nlu.extract("I bought a TV for $500 under ORD-001. I have the receipt.")
        csp_result = self.csp.evaluate(claim, completed_verifications=["evidence_check"])
        risk = self.assessor.assess_risk(
            claim, csp_result, completed_verifications=["evidence_check"]
        )

        self.assertEqual(risk.posterior_details["P(evidence_strong)"], 1.0)
        self.assertEqual(risk.evidence_strength, EvidenceStrength.STRONG)
        self.assertEqual(risk.evidence_reliability, 1.0)
        self.assertLess(risk.fraud_probability, 0.15)
        self.assertTrue(risk.recommendation.startswith("MINIMAL RISK"))

    def test_clean_observation_resetting(self):
        """Verify _clear_observations() resets all nodes between independent assess_risk calls."""
        claim_high_risk = self.nlu.extract(
            "Refund my $500 TV under ORD-001 with this fabricated proof immediately or I will sue!"
        )
        csp_high = self.csp.evaluate(claim_high_risk)
        risk1 = self.assessor.assess_risk(claim_high_risk, csp_high)
        self.assertGreater(risk1.fraud_probability, 0.70)

        # Follow-up with a verified benign claim
        claim_benign = self.nlu.extract("Order ORD-001 refund. Here is my receipt.")
        csp_benign = self.csp.evaluate(claim_benign, completed_verifications=["evidence_check"])
        risk2 = self.assessor.assess_risk(claim_benign, csp_benign, ["evidence_check"])

        self.assertLess(risk2.fraud_probability, 0.15)
        self.assertEqual(risk2.evidence_strength, EvidenceStrength.STRONG)

    def test_file_driven_pipeline_end_to_end(self):
        """Test file-to-file pipeline: claim JSON -> CSP JSON -> RiskAssessment JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claim_file = Path(tmpdir) / "claim.json"
            csp_file = Path(tmpdir) / "csp_out.json"
            risk_file = Path(tmpdir) / "risk_out.json"

            # 1. Write structured claim from NLU to file
            claim = self.nlu.extract("I bought a TV for $500 under ORD-001. I have the receipt.")
            claim.to_file(claim_file)
            self.assertTrue(claim_file.exists())

            # 2. Module 1 CSP takes claim file and outputs csp_out.json
            csp_res = self.csp.evaluate_file(claim_file, output_file=csp_file)
            self.assertTrue(csp_file.exists())
            self.assertEqual(len(csp_res.applicable_policies), 1)

            # 3. Module 3 takes claim file + csp file and outputs risk_out.json
            risk_res = self.assessor.assess_risk_file(
                claim_file=claim_file,
                csp_file=csp_file,
                output_file=risk_file,
            )
            self.assertTrue(risk_file.exists())
            self.assertIsInstance(risk_res, RiskAssessment)

            # 4. Verify risk_out.json can be reloaded and matches exact posteriors
            reloaded_risk = RiskAssessment.from_file(risk_file)
            self.assertEqual(reloaded_risk.fraud_probability, risk_res.fraud_probability)
            self.assertEqual(reloaded_risk.evidence_strength, risk_res.evidence_strength)
            self.assertEqual(reloaded_risk.recommendation, risk_res.recommendation)

    def test_orchestrator_process_claim_file(self):
        """Test full orchestrator file runner: produces csp_result.json, risk_assessment.json, pipeline_response.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            claim_file = Path(tmpdir) / "input_claim.json"
            claim = self.nlu.extract("Order ORD-001 refund. Here is my receipt.")
            claim.to_file(claim_file)

            out_dir = Path(tmpdir) / "pipeline_run"
            trace = self.orchestrator.process_claim_file(claim_file, output_dir=out_dir)

            # Assert all intermediate and final files exist
            self.assertTrue((out_dir / "csp_result.json").exists())
            self.assertTrue((out_dir / "risk_assessment.json").exists())
            self.assertTrue((out_dir / "pipeline_response.json").exists())

            # Check that the saved risk assessment contains valid Bayesian probabilities
            saved_risk = RiskAssessment.from_file(out_dir / "risk_assessment.json")
            self.assertGreaterEqual(saved_risk.fraud_probability, 0.0)
            self.assertLessEqual(saved_risk.fraud_probability, 1.0)
            self.assertIn("P(legitimate)", saved_risk.posterior_details)


if __name__ == "__main__":
    unittest.main()
