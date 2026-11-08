# Module 4 — Customer Support Contextual Bandit Simulation Environment
# ======================================================================
# Custom Gymnasium-compatible simulation environment for training and evaluating
# the Contextual Bandit strategic decision-maker.
#
# Generates realistic synthetic customer cases using hidden Ground Truth,
# executes logic modules (NLU, CSP, Bayes) to build 20-D context vectors,
# and computes scalar rewards based on the exact reward schema.

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from module1.csp import CSPSolver
from module1.policy_graph import PolicyGraph
from module3.bayesian import BayesianRiskAssessor
from module4.context_vector import build_context_vector, CONTEXT_DIMENSION
from module4.contextual_bandit import BANDIT_ARMS, LinUCBBandit
from module4.nlu import NLUExtractor
from shared.models import (
    ActionType, ClaimExtraction, ConstraintStatus, ConversationState,
    EvidenceStrength, ThreatLevel, Sentiment, RiskAssessment, CSPResult
)


@dataclass
class GroundTruthCase:
    """The hidden reality of a simulated customer claim (secret to the environment)."""
    case_id: str
    is_fraud: bool
    has_real_receipt: bool
    is_within_window: bool
    order_exists: bool
    policy_type: str              # "P_RETURN", "P_REFUND", "P_EXCHANGE"
    amount: float
    days_since: int
    persona: str                  # "cooperative", "threat", "emotional", "fabricated"
    item: str
    order_id: str


class CustomerSupportBanditEnv:
    """
    Gymnasium-compatible simulation environment.
    Generates hidden ground-truth customer claims, compiles symptoms into
    20-dimensional context vectors, and dispenses objective scalar rewards.
    """

    MAX_TURNS = 4

    def __init__(
        self,
        fraud_ratio: float = 0.25,
        adversarial_ratio: float = 0.30,
        seed: Optional[int] = None,
    ):
        self.fraud_ratio = float(fraud_ratio)
        self.adversarial_ratio = float(adversarial_ratio)
        self.rng = random.Random(seed)

        # Core logic engines
        self.policy_graph = PolicyGraph()
        self.csp_solver = CSPSolver(self.policy_graph)
        self.bayes_assessor = BayesianRiskAssessor()
        self.nlu = NLUExtractor()

        # Dynamic episode state
        self.current_case: Optional[GroundTruthCase] = None
        self.current_claim: Optional[ClaimExtraction] = None
        self.current_csp: Optional[CSPResult] = None
        self.current_risk: Optional[RiskAssessment] = None
        self.turn: int = 0
        self.investigation_cost: float = 0.0
        self.completed_verifications: List[str] = []
        self.done: bool = False

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Instantiate a new simulated customer claim and return initial 20-D context vector.
        """
        if seed is not None:
            self.rng = random.Random(seed)

        self.turn = 1
        self.investigation_cost = 0.0
        self.completed_verifications = []
        self.done = False

        # 1. Sample hidden ground truth
        self.current_case = self._generate_ground_truth()

        # 2. Generate observable initial symptoms
        self.current_claim = self._synthesize_initial_claim(self.current_case)

        # 3. Evaluate CSP and Bayesian Risk
        self._evaluate_modules()

        # 4. Compile 20-D Context Vector
        context = self._get_context()
        info = {
            "case_id": self.current_case.case_id,
            "is_fraud": self.current_case.is_fraud,
            "policy": self.current_case.policy_type,
            "amount": self.current_case.amount,
        }
        return context, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute chosen arm (0 to 5) and compute scalar reward against Ground Truth.
        """
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() before step().")

        arm_action = BANDIT_ARMS[int(action)]
        reward = 0.0
        info: Dict[str, Any] = {"action": arm_action.value}

        # ---------------------------------------------------------------------
        # Evaluation of Arms & Reward Computation (Section 4 Schema)
        # ---------------------------------------------------------------------
        has_unsatisfied_constraint = 1 if (
            self.current_csp and (
                any(s == ConstraintStatus.VIOLATED for s in self.current_csp.constraints.values())
                or bool(self.current_csp.violated_policies)
            )
        ) else 0

        is_fraud = self.current_case.is_fraud
        is_valid = (
            not is_fraud
            and self.current_case.order_exists
            and self.current_case.is_within_window
        )

        fraud_prob = self.current_risk.fraud_probability if self.current_risk else 0.50

        # Arm 0: APPROVE
        if arm_action == ActionType.APPROVE:
            self.done = True
            if has_unsatisfied_constraint == 1:
                # -50 Policy Violation penalty
                reward -= 50.0
                info["reason"] = "Policy violation: approved with unsatisfied constraint"

            if is_fraud or not self.current_case.has_real_receipt:
                # -200 Financial Loss penalty
                reward -= 200.0
                info["reason"] = "Financial loss: approved fraudulent claim"
            elif is_valid and has_unsatisfied_constraint == 0:
                # +100 Correct Resolution
                reward += 100.0
                info["reason"] = "Correct resolution: approved valid claim"
            else:
                reward -= 20.0

        # Arm 1: REJECT
        elif arm_action == ActionType.REJECT:
            self.done = True
            if is_fraud or not self.current_case.is_within_window or not self.current_case.order_exists:
                # +100 Correct Resolution
                reward += 100.0
                info["reason"] = "Correct resolution: rejected fraudulent/ineligible claim"
            elif is_valid:
                # -100 Unnecessary Rejection of honest customer
                reward -= 100.0
                info["reason"] = "Unnecessary rejection of valid customer"

        # Arm 4: PROPOSE_ALTERNATIVE
        elif arm_action == ActionType.OFFER_ALTERNATIVE:
            self.done = True
            if not self.current_case.is_within_window and not is_fraud:
                # Good compromise for expired window honest customer
                reward += 50.0
                info["reason"] = "Good compromise: store credit for expired window"
            elif is_fraud:
                # -100 offering credit to scammer
                reward -= 100.0
                info["reason"] = "Credit offered to fraudulent user"
            else:
                # Valid customer entitled to full refund, mild CSAT hit
                reward += 20.0
                info["reason"] = "Offered alternative to valid customer"

        # Arm 5: ESCALATE
        elif arm_action == ActionType.ESCALATE:
            self.done = True
            if fraud_prob > 0.90 or fraud_prob < 0.10:
                # -20 Inappropriate Escalation when decision is obvious
                reward -= 20.0
                info["reason"] = "Inappropriate escalation: clear fraud profile"
            elif self.current_case.amount > 500 and not is_fraud:
                # Appropriate high-value escalation
                reward += 40.0
                info["reason"] = "Appropriate escalation for high-value claim"
            else:
                # Standard administrative escalation overhead
                reward -= 10.0
                info["reason"] = "Routine escalation overhead"

        # Arm 2: ASK_EVIDENCE
        elif arm_action == ActionType.ASK_FOR_EVIDENCE:
            self.investigation_cost += 1.0
            reward -= 1.0  # -1 investigation cost
            self.turn += 1

            # Simulate customer reaction
            if self.current_case.has_real_receipt:
                # Honest customer submits genuine receipt & photo
                if "receipt" not in self.current_claim.evidence_provided:
                    self.current_claim.evidence_provided.append("receipt")
                if "photo" not in self.current_claim.evidence_provided:
                    self.current_claim.evidence_provided.append("photo")
                if "evidence_check" not in self.completed_verifications:
                    self.completed_verifications.append("evidence_check")
            else:
                # Fraudulent customer either refuses or submits fabricated evidence
                if "fabricated_evidence" not in self.current_claim.adversarial_tactics:
                    self.current_claim.adversarial_tactics.append("fabricated_evidence")

            self._evaluate_modules()
            if self.turn >= self.MAX_TURNS:
                self.done = True

        # Arm 3: PERFORM_VERIFICATION
        elif arm_action == ActionType.INVESTIGATE:
            self.investigation_cost += 1.0
            reward -= 1.0  # -1 investigation cost
            self.turn += 1

            if "order_lookup" not in self.completed_verifications and self.current_case.order_exists:
                self.completed_verifications.append("order_lookup")
            if "history_check" not in self.completed_verifications:
                self.completed_verifications.append("history_check")

            self._evaluate_modules()
            if self.turn >= self.MAX_TURNS:
                self.done = True

        next_context = self._get_context()
        truncated = False
        return next_context, reward, self.done, truncated, info

    def _generate_ground_truth(self) -> GroundTruthCase:
        """Sample independent hidden attributes."""
        is_fraud = self.rng.random() < self.fraud_ratio
        policy_type = self.rng.choice(["P_RETURN", "P_REFUND", "P_EXCHANGE"])

        # Amounts: skewed log-uniform
        tier = self.rng.choice(["low", "mid", "high", "extreme"])
        if tier == "low":
            amount = round(self.rng.uniform(20.0, 99.0), 2)
        elif tier == "mid":
            amount = round(self.rng.uniform(100.0, 250.0), 2)
        elif tier == "high":
            amount = round(self.rng.uniform(251.0, 600.0), 2)
        else:
            amount = round(self.rng.uniform(601.0, 1400.0), 2)

        is_within_window = not is_fraud if self.rng.random() < 0.85 else False
        days_since = self.rng.randint(2, 25) if is_within_window else self.rng.randint(35, 65)

        has_real_receipt = (not is_fraud) and (self.rng.random() < 0.80)
        order_exists = not (is_fraud and self.rng.random() < 0.40)

        # Persona sampling
        if is_fraud or self.rng.random() < self.adversarial_ratio:
            persona = self.rng.choice(["threat", "emotional", "fabricated"])
        else:
            persona = "cooperative"

        item = self.rng.choice(["TV", "Headphones", "Camera", "Laptop", "Shoes", "Watch"])
        order_id = self.rng.choice(["ORD-001", "ORD-002", "ORD-003", "ORD-004"]) if order_exists else f"ORD-{self.rng.randint(900, 999)}"

        return GroundTruthCase(
            case_id=f"CASE_{self.rng.randint(10000, 99999)}",
            is_fraud=is_fraud,
            has_real_receipt=has_real_receipt,
            is_within_window=is_within_window,
            order_exists=order_exists,
            policy_type=policy_type,
            amount=amount,
            days_since=days_since,
            persona=persona,
            item=item,
            order_id=order_id,
        )

    def _synthesize_initial_claim(self, case: GroundTruthCase) -> ClaimExtraction:
        """Synthesize initial customer claim based on persona and ground truth."""
        p_date = (datetime.now() - timedelta(days=case.days_since)).strftime("%Y-%m-%d")
        tactics = []
        threats = []
        emotional_appeals = []
        persuasive = []
        threat_level = ThreatLevel.NONE
        sentiment = Sentiment.COOPERATIVE

        # Starting evidence
        evidence_provided = []
        if case.has_real_receipt and self.rng.random() < 0.50:
            evidence_provided.append("receipt")

        if case.persona == "threat":
            threat_level = ThreatLevel.HIGH
            sentiment = Sentiment.THREATENING
            tactics.extend(["legal_threat", "chargeback_threat"])
            threats.extend(["legal_action", "chargeback"])
            persuasive.append("threat_intimidation")
            raw_text = f"Refund my ${case.amount:.2f} {case.item} under {case.order_id} right now or I will sue and file a bank chargeback!"
        elif case.persona == "emotional":
            threat_level = ThreatLevel.MEDIUM
            sentiment = Sentiment.FRUSTRATED
            tactics.append("emotional_appeal")
            emotional_appeals.extend(["single_parent_family", "medical_hardship"])
            persuasive.append("sympathy_exploitation")
            raw_text = f"Please help me, I am a single parent and need my ${case.amount:.2f} back for order {case.order_id} to buy medicine for my sick child."
        elif case.persona == "fabricated":
            threat_level = ThreatLevel.HIGH
            sentiment = Sentiment.ANGRY
            tactics.extend(["fabricated_evidence", "contradiction"])
            raw_text = f"I bought this {case.item} yesterday for ${case.amount:.2f} (Order {case.order_id}), but the box is completely unopened and the screen inside is shattered. Refund me immediately!"
        else:
            raw_text = f"Hi, I would like a {case.policy_type.lower().replace('p_', '')} for my {case.item} (${case.amount:.2f}) purchased under {case.order_id} on {p_date}."

        intent_map = {"P_RETURN": "return", "P_REFUND": "refund", "P_EXCHANGE": "exchange"}
        return ClaimExtraction(
            intent=intent_map.get(case.policy_type, "refund"),
            amount=case.amount,
            item=case.item,
            purchase_date=p_date,
            order_id=case.order_id,
            evidence_provided=evidence_provided,
            threat_level=threat_level,
            sentiment=sentiment,
            adversarial_tactics=tactics,
            raw_text=raw_text,
            claimed_previous_claims=None,
            contradictory_claims=[{"type": "claim_statement_conflict", "snippet": "unopened but shattered"}] if "contradiction" in tactics else [],
            supporting_evidence=[],
            emotional_appeals=emotional_appeals,
            threats=threats,
            persuasive_strategies=persuasive,
            conversation_history=[{"role": "user", "content": raw_text}],
        )

    def _evaluate_modules(self):
        """Invoke Module 1 (CSP) and Module 3 (Bayes) on current claim and verifications."""
        self.current_csp = self.csp_solver.evaluate(
            self.current_claim,
            completed_verifications=self.completed_verifications
        )
        self.current_risk = self.bayes_assessor.assess_risk(
            self.current_claim,
            self.current_csp,
            completed_verifications=self.completed_verifications
        )

    def _get_context(self) -> np.ndarray:
        """Build normalized 20-D context vector."""
        return build_context_vector(
            claim=self.current_claim,
            csp_result=self.current_csp,
            risk_assessment=self.current_risk,
            interaction_turns=self.turn,
            investigation_cost=self.investigation_cost,
        )
