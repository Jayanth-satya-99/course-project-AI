# Tests for Module 4: 54,000-State Tabular Q-Learning Implementation
# ===================================================================

import os
import unittest
import numpy as np

from shared.models import (
    ActionType, ClaimExtraction, ConstraintStatus, ConversationState,
    EvidenceStrength, ThreatLevel, Sentiment, RiskAssessment, CSPResult
)
from module1.policy_graph import PolicyGraph
from module1.csp import CSPSolver
from module3.bayesian import BayesianRiskAssessor
from module4.rl_agent import RLAgent
from module4.generate_q_dataset import (
    TOTAL_STATES, NUM_ACTIONS, encode_state, decode_state,
    encode_constraints, decode_constraints, calculate_reward,
    ACTION_APPROVE, ACTION_REJECT, ACTION_ASK_FOR_EVIDENCE,
    ACTION_PERFORM_VERIFICATION, ACTION_PROPOSE_ALTERNATIVE, ACTION_ESCALATE
)


class Module4QLearningTests(unittest.TestCase):
    """Test suite for the 54,000-state Tabular Q-Learning system."""

    def setUp(self):
        self.rl_agent = RLAgent()
        self.csp_solver = CSPSolver(PolicyGraph())
        self.bayes_assessor = BayesianRiskAssessor()

    def test_state_encoding_and_decoding_bijectivity(self):
        """Assert state encoding and decoding are perfectly bijective across boundaries."""
        # Minimum boundary: 0
        min_state = encode_state(0, 0, 0, 0, 0)
        self.assertEqual(min_state, 0)
        self.assertEqual(decode_state(0), (0, 0, 0, 0, 0))

        # Maximum boundary: 53,999
        max_state = encode_state(4, 26, 3, 9, 9)
        self.assertEqual(max_state, 53999)
        self.assertEqual(decode_state(53999), (4, 26, 3, 9, 9))

        # Example 1 from specification: policy=0, cs=26, ev=3, turn=0, fb=2 -> 10702
        ex1 = encode_state(0, 26, 3, 0, 2)
        self.assertEqual(ex1, 10702)
        self.assertEqual(decode_state(10702), (0, 26, 3, 0, 2))

        # Test base-3 constraint encoding
        self.assertEqual(encode_constraints(2, 2, 2), 26)  # All satisfied
        self.assertEqual(encode_constraints(1, 1, 1), 13)  # All uncertain
        self.assertEqual(encode_constraints(0, 0, 0), 0)   # All violated
        self.assertEqual(decode_constraints(26), (2, 2, 2))
        self.assertEqual(decode_constraints(13), (1, 1, 1))
        self.assertEqual(decode_constraints(0), (0, 0, 0))

    def test_reward_function_exact_business_rules(self):
        """Assert that calculate_reward adheres to the exact specified business rewards."""
        # 1. Action 0: APPROVE
        # Genuine compliant -> +100
        r_app_gen = calculate_reward(0, 26, 3, 0, 1, ACTION_APPROVE, ground_truth_fraud=0)
        self.assertEqual(r_app_gen, 100.0)

        # Fraud approval (Financial loss) -> -200
        r_app_fraud = calculate_reward(0, 26, 3, 0, 8, ACTION_APPROVE, ground_truth_fraud=1)
        self.assertEqual(r_app_fraud, -200.0)

        # Genuine with constraint violation (Policy violation) -> -50
        cs_violated = encode_constraints(2, 0, 2)  # c2 violated
        r_app_vio = calculate_reward(0, cs_violated, 3, 0, 1, ACTION_APPROVE, ground_truth_fraud=0)
        self.assertEqual(r_app_vio, -50.0)

        # 2. Action 1: REJECT
        # Fraud rejection (Correct resolution) -> +100
        r_rej_fraud = calculate_reward(0, 0, 2, 0, 9, ACTION_REJECT, ground_truth_fraud=1)
        self.assertEqual(r_rej_fraud, 100.0)

        # Genuine rejection (Unnecessary rejection) -> -100
        r_rej_gen = calculate_reward(0, 26, 3, 0, 1, ACTION_REJECT, ground_truth_fraud=0)
        self.assertEqual(r_rej_gen, -100.0)

        # 3. Actions 2 & 3: Investigation steps (-1) and timeout (-25)
        r_ask = calculate_reward(0, 13, 0, 1, 4, ACTION_ASK_FOR_EVIDENCE, ground_truth_fraud=0)
        self.assertEqual(r_ask, -1.0)
        r_ver = calculate_reward(0, 13, 0, 1, 4, ACTION_PERFORM_VERIFICATION, ground_truth_fraud=0)
        self.assertEqual(r_ver, -1.0)
        r_timeout = calculate_reward(0, 13, 0, 9, 4, ACTION_ASK_FOR_EVIDENCE, ground_truth_fraud=0, timed_out=True)
        self.assertEqual(r_timeout, -25.0)

        # 4. Action 4: PROPOSE_ALTERNATIVE (+60 genuine, -80 fraud)
        r_alt_gen = calculate_reward(0, cs_violated, 3, 1, 1, ACTION_PROPOSE_ALTERNATIVE, ground_truth_fraud=0)
        self.assertEqual(r_alt_gen, 60.0)
        r_alt_fraud = calculate_reward(0, cs_violated, 2, 1, 8, ACTION_PROPOSE_ALTERNATIVE, ground_truth_fraud=1)
        self.assertEqual(r_alt_fraud, -80.0)

        # 5. Action 5: ESCALATE (+35 uncertain bucket 3-7, -20 obvious)
        r_esc_unc = calculate_reward(0, 13, 1, 5, 5, ACTION_ESCALATE, ground_truth_fraud=0)
        self.assertEqual(r_esc_unc, 35.0)
        r_esc_low = calculate_reward(0, 26, 3, 1, 1, ACTION_ESCALATE, ground_truth_fraud=0)
        self.assertEqual(r_esc_low, -20.0)
        r_esc_high = calculate_reward(0, 0, 2, 1, 9, ACTION_ESCALATE, ground_truth_fraud=1)
        self.assertEqual(r_esc_high, -20.0)

    def test_q_table_loading_and_dimensions(self):
        """Assert the trained Q-table has shape (54000, 6) and is non-empty."""
        self.assertIsNotNone(self.rl_agent.q_table_54k)
        self.assertEqual(self.rl_agent.q_table_54k.shape, (54000, 6))
        # Ensure values are loaded and not all zeros
        self.assertGreater(np.abs(self.rl_agent.q_table_54k).sum(), 1000.0)

    def test_rl_agent_state_encoding_from_conversation(self):
        """Assert RLAgent.encode_state_id properly maps a live ConversationState."""
        conv = ConversationState(session_id="test_sess")
        conv.turn = 2
        conv.claim = ClaimExtraction(
            raw_text="I want a refund for my order ORD-001.",
            intent="refund",
            amount=150.0,
            order_id="ORD-001",
            item="Headphones",
            purchase_date="2026-10-01",
            evidence_provided=["receipt.pdf"],
            threat_level=ThreatLevel.NONE,
            sentiment=Sentiment.COOPERATIVE,
            adversarial_tactics=[],
        )
        conv.completed_verifications = ["order_lookup"]
        csp_res = self.csp_solver.evaluate(conv.claim, conv.completed_verifications)
        conv.csp_result = csp_res
        risk = self.bayes_assessor.assess_risk(conv.claim, csp_res, conv.completed_verifications)
        conv.risk_assessment = risk

        state_id = self.rl_agent.encode_state_id(conv)
        self.assertIsInstance(state_id, int)
        self.assertGreaterEqual(state_id, 0)
        self.assertLess(state_id, TOTAL_STATES)

        p, cs, ev, t, fb = self.rl_agent.decode_state_id(state_id)
        self.assertEqual(p, 0)  # Refund policy
        self.assertEqual(t, 2)  # Turn 2
        self.assertIn(ev, (0, 1, 2, 3))
        self.assertIn(fb, list(range(10)))

    def test_strategic_action_selection_on_key_scenarios(self):
        """Verify the trained Q-table selects strategically optimal actions."""
        # Scenario A: Genuine compliant claim with all constraints satisfied & low risk
        # State: policy=0, cs=26 (all satisfied), ev=3 (strong), turn=0, fb=1
        s_clean = encode_state(0, 26, 3, 0, 1)
        q_clean = self.rl_agent.q_table_54k[s_clean]
        best_clean = np.argmax(q_clean)
        self.assertEqual(best_clean, ACTION_APPROVE, "Compliant claim should be APPROVE")

        # Scenario B: Clear fraud with contradictory evidence & high risk
        # State: policy=0, cs=0 (all violated), ev=2 (contradictory), turn=0, fb=9
        s_fraud = encode_state(0, 0, 2, 0, 9)
        q_fraud = self.rl_agent.q_table_54k[s_fraud]
        best_fraud = np.argmax(q_fraud)
        self.assertEqual(best_fraud, ACTION_REJECT, "Clear fraud should be REJECT")

        # Scenario C: Genuine customer past policy return window -> Alternative Offer
        # State: policy=0, cs=encode_constraints(2, 0, 2), ev=3, turn=1, fb=1
        s_alt = encode_state(0, encode_constraints(2, 0, 2), 3, 1, 1)
        q_alt = self.rl_agent.q_table_54k[s_alt]
        best_alt = np.argmax(q_alt)
        self.assertEqual(best_alt, ACTION_PROPOSE_ALTERNATIVE, "Expired window genuine should be PROPOSE_ALTERNATIVE")

        # Scenario D: Lingering uncertainty at late turn (Turn 8, bucket 5)
        # State: policy=0, cs=13 (uncertain), ev=1, turn=8, fb=5
        s_late = encode_state(0, 13, 1, 8, 5)
        q_late = self.rl_agent.q_table_54k[s_late]
        best_late = np.argmax(q_late)
        self.assertEqual(best_late, ACTION_ESCALATE, "Lingering late-turn ambiguity should be ESCALATE")


if __name__ == "__main__":
    unittest.main()

