import os
import tempfile
import unittest
import numpy as np

from module1.csp import CSPSolver
from module1.policy_graph import PolicyGraph
from module3.bayesian import BayesianRiskAssessor
from module4.context_vector import (
    build_context_vector, extract_nlu_perception_features,
    CONTEXT_DIMENSION, FEATURE_NAMES
)
from module4.contextual_bandit import LinUCBBandit, BANDIT_ARMS
from module4.environment import CustomerSupportBanditEnv, GroundTruthCase
from module4.nlu import NLUExtractor
from module4.rl_agent import RLAgent
from module4.train_bandit import (
    Baseline1Escaler, Baseline2Accommodator, evaluate_agent
)
from shared.models import (
    ActionType, ClaimExtraction, ConstraintStatus, ConversationState,
    EvidenceStrength, ThreatLevel, Sentiment, RiskAssessment, CSPResult
)


class Module4ContextualBanditTests(unittest.TestCase):
    def setUp(self):
        self.nlu = NLUExtractor()
        self.csp = CSPSolver(PolicyGraph())
        self.bayes = BayesianRiskAssessor()
        self.rl_agent = RLAgent()

    def test_perception_layer_extraction(self):
        """Test extraction of perception layer variables and 4 adversarial flags."""
        text = (
            "Refund my $500 TV (Order ORD-001) immediately or I will file a bank chargeback! "
            "Please understand I am a single mother. The package was factory sealed and unopened but the screen inside is completely cracked. "
            "Your website policy clearly promised no-hassle refunds."
        )
        claim = self.nlu.extract(text)
        features = extract_nlu_perception_features(claim)

        self.assertEqual(features["claim_amount"], 500.0)
        self.assertIsInstance(features["days_since_transaction"], int)
        self.assertIn(features["has_provided_evidence"], (0, 1))

        flags = features["adversarial_flags"]
        self.assertIn("threat", flags)
        self.assertIn("emotional_appeal", flags)
        self.assertIn("contradiction", flags)
        self.assertIn("persuasive_strategy", flags)

        # Assert all 4 flags were triggered by the prompt
        self.assertEqual(flags["threat"], 1)
        self.assertEqual(flags["emotional_appeal"], 1)
        self.assertEqual(flags["contradiction"], 1)
        self.assertEqual(flags["persuasive_strategy"], 1)

    def test_context_vector_dimensions_and_normalization(self):
        """Assert context vector is exactly 20-dimensional with all features in [0.0, 1.0]."""
        claim = self.nlu.extract("I bought headphones for $79.99 under ORD-002. I have the receipt.")
        csp_res = self.csp.evaluate(claim)
        risk = self.bayes.assess_risk(claim, csp_res)

        context = build_context_vector(
            claim=claim,
            csp_result=csp_res,
            risk_assessment=risk,
            interaction_turns=2,
            investigation_cost=2.0,
        )

        self.assertIsInstance(context, np.ndarray)
        self.assertEqual(context.shape, (CONTEXT_DIMENSION,))
        self.assertEqual(len(FEATURE_NAMES), CONTEXT_DIMENSION)

        # Check normalization bounds
        for i, val in enumerate(context):
            self.assertGreaterEqual(val, 0.0, f"Feature {FEATURE_NAMES[i]} < 0.0 ({val})")
            self.assertLessEqual(val, 1.0, f"Feature {FEATURE_NAMES[i]} > 1.0 ({val})")

    def test_linucb_bandit_selection_and_updates(self):
        """Test LinUCB arm selection, Sherman-Morrison rank-1 update, and weight persistence."""
        bandit = LinUCBBandit(n_features=CONTEXT_DIMENSION, n_arms=6, alpha=0.50)
        context = np.random.uniform(0.0, 1.0, size=(CONTEXT_DIMENSION,))

        # Arm selection
        arm, scores = bandit.select_arm(context, explore=True)
        self.assertIn(arm, list(range(6)))
        self.assertEqual(scores.shape, (6,))

        # Test valid masking (e.g., allow only arm 1 and 2)
        masked_arm, masked_scores = bandit.select_arm(context, valid_arms=[1, 2], explore=True)
        self.assertIn(masked_arm, [1, 2])

        # Test online update
        initial_count = bandit.arm_counts[arm]
        bandit.update(arm, context, reward=100.0)
        self.assertEqual(bandit.arm_counts[arm], initial_count + 1)
        self.assertEqual(bandit.arm_rewards[arm], 100.0)

        # Test save and load weights
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_file = os.path.join(tmpdir, "test_weights.npz")
            bandit.save_weights(weights_file)
            self.assertTrue(os.path.exists(weights_file))

            loaded_bandit = LinUCBBandit(n_features=CONTEXT_DIMENSION, n_arms=6)
            loaded_bandit.load_weights(weights_file)
            self.assertEqual(loaded_bandit.arm_counts[arm], bandit.arm_counts[arm])
            np.testing.assert_allclose(loaded_bandit.b[arm], bandit.b[arm])

    def test_simulation_environment_exact_rewards(self):
        """Verify the exact reward values: +100, -200, -100, -50, -20, -1."""
        env = CustomerSupportBanditEnv(seed=123)
        context, info = env.reset()

        self.assertEqual(context.shape, (CONTEXT_DIMENSION,))
        self.assertFalse(env.done)

        # 1. Test investigation cost penalty (-1)
        _, reward_evidence, done_evidence, _, _ = env.step(2)  # ASK_EVIDENCE
        self.assertEqual(reward_evidence, -1.0)
        self.assertFalse(done_evidence)

        # 2. Test manual ground truth case evaluation for exact reward signals
        # Scenario A: Fraud claim approved -> -200 financial loss
        env.current_case = GroundTruthCase(
            case_id="TEST_FRAUD",
            is_fraud=True,
            has_real_receipt=False,
            is_within_window=True,
            order_exists=True,
            policy_type="P_REFUND",
            amount=500.0,
            days_since=5,
            persona="fabricated",
            item="TV",
            order_id="ORD-001",
        )
        env.done = False
        _, reward_loss, done_loss, _, info_loss = env.step(0)  # APPROVE
        self.assertTrue(done_loss)
        self.assertIn(reward_loss, (-200.0, -250.0))  # -200 loss (and optional -50 if constraint unsatisfied)

        # Scenario B: Fraud claim rejected -> +100 correct resolution
        env.done = False
        _, reward_reject_fraud, done_rf, _, _ = env.step(1)  # REJECT
        self.assertEqual(reward_reject_fraud, 100.0)
        self.assertTrue(done_rf)

        # Scenario C: Honest valid claim rejected -> -100 unnecessary rejection
        env.current_case = GroundTruthCase(
            case_id="TEST_VALID",
            is_fraud=False,
            has_real_receipt=True,
            is_within_window=True,
            order_exists=True,
            policy_type="P_REFUND",
            amount=100.0,
            days_since=5,
            persona="cooperative",
            item="Headphones",
            order_id="ORD-002",
        )
        env.done = False
        _, reward_false_reject, done_fr, _, _ = env.step(1)  # REJECT
        self.assertEqual(reward_false_reject, -100.0)

        # Scenario D: Inappropriate escalation when fraud probability is extreme -> -20
        env.done = False
        env.current_risk.fraud_probability = 0.95
        _, reward_inapp_esc, done_esc, _, _ = env.step(5)  # ESCALATE
        self.assertEqual(reward_inapp_esc, -20.0)

    def test_bandit_outperforms_baselines(self):
        """Assert that the trained LinUCB Bandit mathematically outperforms Baseline 1 and Baseline 2."""
        weights_path = os.path.join(os.path.dirname(__file__), "..", "module4", "bandit_weights.npz")
        self.assertTrue(os.path.exists(weights_path), "bandit_weights.npz should exist from training")

        bandit = LinUCBBandit(n_features=CONTEXT_DIMENSION, n_arms=6)
        bandit.load_weights(weights_path)

        env1 = CustomerSupportBanditEnv(seed=777)
        env2 = CustomerSupportBanditEnv(seed=777)
        env3 = CustomerSupportBanditEnv(seed=777)

        escaler = Baseline1Escaler()
        accommodator = Baseline2Accommodator()

        res_bandit = evaluate_agent(bandit, env1, test_episodes=200, is_bandit=True)
        res_base1 = evaluate_agent(escaler, env2, test_episodes=200, is_bandit=False)
        res_base2 = evaluate_agent(accommodator, env3, test_episodes=200, is_bandit=False)

        # Bandit should have higher cumulative reward than both baselines
        self.assertGreater(res_bandit["total_reward"], res_base1["total_reward"])
        self.assertGreater(res_bandit["total_reward"], res_base2["total_reward"])

        # Baseline 1 has financial losses / severe penalties; Bandit should maintain lower financial loss rate
        self.assertLessEqual(res_bandit["financial_loss_rate"], res_base1["financial_loss_rate"])


if __name__ == "__main__":
    unittest.main()
