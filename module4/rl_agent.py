# Module 4 — Reinforcement Learning Agent (The Grand Strategist)
# ==============================================================
# Implements the strategic decision-making loop for the Policy-Adherent Agent.
#
# In Pattern 2 (RL-Centric Orchestrator Hub), Module 4 acts as the autonomous
# commander. Its action space includes:
#   1. Internal Epistemic Actions: CALL_CSP, CALL_BAYES
#   2. External Communicative Actions: ASK_FOR_EVIDENCE, INVESTIGATE,
#      OFFER_ALTERNATIVE, ESCALATE, APPROVE, REJECT
#
# State Space formulation:
#   s = (turn, amount_tier, threat_level, csp_evaluated, num_satisfied,
#        num_violated, num_uncertain, num_manipulated, bayes_evaluated,
#        fraud_tier, evidence_tier)
#
# Reward Function:
#   R = R_resolution + R_compliance - C_investigation - P_financial_loss - P_false_rejection

import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from shared.models import (
    ActionType, ClaimExtraction, CSPResult, RiskAssessment,
    ConstraintStatus, ThreatLevel, EvidenceStrength, ConversationState
)
from module4.contextual_bandit import LinUCBBandit, BANDIT_ARMS
from module4.context_vector import build_context_vector, CONTEXT_DIMENSION

logger = logging.getLogger(__name__)


class RLAgent:
    """
    Q-Learning / Contextual Bandit Agent serving as the strategic orchestrator.
    Determines whether to query analytical tools (CSP, Bayes) or execute external
    macro actions (Ask Evidence, Investigate, Escalate, Approve, Reject).
    """

    EXTERNAL_ACTIONS = [
        ActionType.APPROVE,
        ActionType.REJECT,
        ActionType.ASK_FOR_EVIDENCE,
        ActionType.INVESTIGATE,
        ActionType.ESCALATE,
        ActionType.OFFER_ALTERNATIVE,
    ]

    Q_ARMS = [
        ActionType.APPROVE,              # Arm 0 (Terminal)
        ActionType.REJECT,               # Arm 1 (Terminal)
        ActionType.ASK_FOR_EVIDENCE,     # Arm 2 (Non-terminal)
        ActionType.INVESTIGATE,          # Arm 3 (Non-terminal)
        ActionType.OFFER_ALTERNATIVE,    # Arm 4 (Terminal)
        ActionType.ESCALATE,             # Arm 5 (Terminal)
    ]

    ARM_TO_ACTION = {
        0: ActionType.APPROVE,
        1: ActionType.REJECT,
        2: ActionType.ASK_FOR_EVIDENCE,
        3: ActionType.INVESTIGATE,
        4: ActionType.OFFER_ALTERNATIVE,
        5: ActionType.ESCALATE,
    }

    ACTION_TO_ARM = {
        ActionType.APPROVE: 0,
        ActionType.REJECT: 1,
        ActionType.ASK_FOR_EVIDENCE: 2,
        ActionType.INVESTIGATE: 3,
        ActionType.OFFER_ALTERNATIVE: 4,
        ActionType.ESCALATE: 5,
    }

    ALL_ACTIONS = [
        ActionType.CALL_CSP,
        ActionType.CALL_BAYES,
        ActionType.ASK_FOR_EVIDENCE,
        ActionType.INVESTIGATE,
        ActionType.OFFER_ALTERNATIVE,
        ActionType.ESCALATE,
        ActionType.APPROVE,
        ActionType.REJECT,
    ]

    def __init__(self, learning_rate: float = 0.15, discount_factor: float = 0.90,
                 epsilon: float = 0.0, q_table_path: Optional[str] = None,
                 bandit_weights_path: Optional[str] = None,
                 q_table_54k_path: Optional[str] = None):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.q_table_path = q_table_path or os.path.join(
            os.path.dirname(__file__), "q_table.json"
        )
        self._load_or_init_q_table()

        # 54k Tabular Q-Table Engine (Idea 1)
        self.q_table_54k: Optional[np.ndarray] = None
        self.q_table_54k_path = q_table_54k_path or os.path.join(
            os.path.dirname(__file__), "q_table_54k.npy"
        )
        if os.path.exists(self.q_table_54k_path):
            try:
                self.q_table_54k = np.load(self.q_table_54k_path)
                logger.info("Loaded 54k Q-table from %s", self.q_table_54k_path)
            except Exception as exc:
                logger.warning("Could not load 54k Q-table: %s", exc)

        # Contextual Bandit (LinUCB) Engine
        self.bandit = LinUCBBandit(n_features=CONTEXT_DIMENSION, n_arms=6, alpha=0.50)
        self.bandit_weights_path = bandit_weights_path or os.path.join(
            os.path.dirname(__file__), "bandit_weights.npz"
        )
        if os.path.exists(self.bandit_weights_path):
            try:
                self.bandit.load_weights(self.bandit_weights_path)
            except Exception as exc:
                logger.warning("Could not load bandit weights: %s", exc)

    @staticmethod
    def encode_state_id(
        conv_state: Optional[ConversationState] = None,
        policy_id: Optional[int] = None,
        constraint_state: Optional[int] = None,
        evidence_rel: Optional[int] = None,
        turn: Optional[int] = None,
        fraud_prob_bucket: Optional[int] = None,
    ) -> int:
        """
        Encode discrete state into state_id in [0, 53999]:
          state_id = (((policy_id * 27 + constraint_state) * 4 + evidence_rel) * 10 + turn) * 10 + fraud_prob_bucket
        Can be called with (conv_state) or individual components.
        """
        if conv_state is not None:
            claim = conv_state.claim
            csp = conv_state.csp_result
            risk = conv_state.risk_assessment

            # 1. Policy ID (0..4)
            if policy_id is None:
                intent = (claim.intent.lower() if claim and claim.intent else "")
                policy_names = " ".join(csp.applicable_policies).lower() if csp else ""
                combined = f"{intent} {policy_names}"

                if "refund" in combined:
                    policy_id = 0
                elif any(k in combined for k in ("return", "exchange", "replacement")):
                    policy_id = 1
                elif any(k in combined for k in ("warranty", "defect", "insurance")):
                    policy_id = 2
                elif any(k in combined for k in ("cancel", "price_match", "payment", "order_modification")):
                    policy_id = 3
                elif any(k in combined for k in ("delivery", "shipment", "missing", "courier")):
                    policy_id = 4
                else:
                    policy_id = 0

            # 2. Constraint State (0..26)
            if constraint_state is None:
                c1, c2, c3 = 1, 1, 1
                if csp is not None and csp.constraints:
                    c1_statuses = []
                    c2_statuses = []
                    c3_statuses = []
                    for k, s in csp.constraints.items():
                        kl = k.lower()
                        if any(w in kl for w in ("order", "record", "payment", "account", "c1")):
                            c1_statuses.append(s)
                        elif any(w in kl for w in ("window", "eligibility", "deadline", "days", "status", "c2")):
                            c2_statuses.append(s)
                        elif any(w in kl for w in ("evidence", "receipt", "photo", "proof", "document", "c3")):
                            c3_statuses.append(s)

                    def _resolve_status(statuses, default_val=1):
                        if not statuses:
                            return default_val
                        if any(s in (ConstraintStatus.VIOLATED, ConstraintStatus.POTENTIALLY_MANIPULATED) for s in statuses):
                            return 0
                        if all(s == ConstraintStatus.SATISFIED for s in statuses):
                            return 2
                        return 1

                    c1 = _resolve_status(c1_statuses, 2 if "order_lookup" in conv_state.completed_verifications else 1)
                    c2 = _resolve_status(c2_statuses, 2)
                    c3 = _resolve_status(c3_statuses, 2 if (claim and claim.evidence_provided) else 1)
                else:
                    if "order_lookup" in conv_state.completed_verifications:
                        c1 = 2
                    if claim and claim.evidence_provided:
                        c3 = 2

                constraint_state = (c1 * 9) + (c2 * 3) + c3

            # 3. Evidence Reliability (0..3)
            if evidence_rel is None:
                if risk is not None:
                    ev_map = {
                        EvidenceStrength.MISSING: 0,
                        EvidenceStrength.WEAK: 1,
                        EvidenceStrength.CONTRADICTORY: 2,
                        EvidenceStrength.FABRICATED: 2,
                        EvidenceStrength.MODERATE: 3,
                        EvidenceStrength.STRONG: 3,
                    }
                    evidence_rel = ev_map.get(risk.evidence_strength, 1)
                elif claim is not None:
                    if any(t in claim.adversarial_tactics for t in ("contradiction", "fabricated_evidence")):
                        evidence_rel = 2
                    elif claim.evidence_provided:
                        evidence_rel = 3
                    else:
                        evidence_rel = 0
                else:
                    evidence_rel = 0

            # 4. Turn (0..9)
            if turn is None:
                turn = min(9, max(0, conv_state.turn))

            # 5. Fraud Probability Bucket (0..9)
            if fraud_prob_bucket is None:
                if risk is not None:
                    fraud_prob_bucket = min(9, max(0, int(risk.fraud_probability * 10)))
                elif claim is not None and claim.threat_level in (ThreatLevel.HIGH, ThreatLevel.MEDIUM):
                    fraud_prob_bucket = 7
                else:
                    fraud_prob_bucket = 2

        # Clamp all variables to valid ranges
        p = min(4, max(0, int(policy_id if policy_id is not None else 0)))
        cs = min(26, max(0, int(constraint_state if constraint_state is not None else 13)))
        ev = min(3, max(0, int(evidence_rel if evidence_rel is not None else 0)))
        t = min(9, max(0, int(turn if turn is not None else 0)))
        fb = min(9, max(0, int(fraud_prob_bucket if fraud_prob_bucket is not None else 2)))

        return (((p * 27 + cs) * 4 + ev) * 10 + t) * 10 + fb

    @staticmethod
    def decode_state_id(state_id: int) -> Tuple[int, int, int, int, int]:
        """Decode state_id in [0, 53999] into (policy_id, constraint_state, evidence_rel, turn, fraud_prob_bucket)."""
        bucket = state_id % 10
        rem1 = state_id // 10
        turn = rem1 % 10
        rem2 = rem1 // 10
        evidence_rel = rem2 % 4
        rem3 = rem2 // 4
        constraint_state = rem3 % 27
        policy_id = rem3 // 27
        return policy_id, constraint_state, evidence_rel, turn, bucket

    def select_q_action(
        self,
        conv_state: ConversationState,
        valid_actions: Optional[List[ActionType]] = None,
        force_exploitation: bool = True,
    ) -> Tuple[ActionType, int, Dict[str, float]]:
        """
        Select action using the trained 54,000-state Tabular Q-Table.
        Applies action masking over valid actions to enforce business invariants.
        Returns (chosen_action, state_id, q_values_dict).
        """
        state_id = self.encode_state_id(conv_state)

        if valid_actions is None:
            valid_actions = self.get_valid_actions(conv_state, csp_queried=True, bayes_queried=True)

        valid_arms = [self.ACTION_TO_ARM[a] for a in valid_actions if a in self.ACTION_TO_ARM]
        if not valid_arms:
            valid_arms = list(range(6))

        if self.q_table_54k is not None and 0 <= state_id < len(self.q_table_54k):
            q_row = self.q_table_54k[state_id]
        else:
            q_row = np.zeros(6, dtype=np.float32)

        q_dict = {self.ARM_TO_ACTION[i].value: float(q_row[i]) for i in range(6)}

        # Exploration vs Exploitation
        if not force_exploitation and random.random() < self.epsilon:
            chosen_arm = random.choice(valid_arms)
        else:
            best_arm = valid_arms[0]
            best_val = q_row[best_arm]
            for arm in valid_arms[1:]:
                if q_row[arm] > best_val:
                    best_val = q_row[arm]
                    best_arm = arm
            chosen_arm = best_arm

        chosen_action = self.ARM_TO_ACTION[chosen_arm]
        return chosen_action, state_id, q_dict

    def select_bandit_action(
        self,
        conv_state: ConversationState,
        explore: bool = False,
    ) -> Tuple[ActionType, np.ndarray, np.ndarray]:
        """
        Select action using the trained LinUCB Contextual Bandit over the 20-D context vector.
        Returns (chosen_action, context_vector, ucb_scores).
        """
        context = build_context_vector(
            claim=conv_state.claim,
            csp_result=conv_state.csp_result,
            risk_assessment=conv_state.risk_assessment,
            conv_state=conv_state,
        )

        valid_actions = self.get_valid_actions(conv_state, csp_queried=True, bayes_queried=True)
        valid_arms = [self.bandit.action_to_arm(a) for a in valid_actions if a in self.EXTERNAL_ACTIONS]
        if not valid_arms:
            valid_arms = list(range(6))

        arm_idx, scores = self.bandit.select_arm(context, valid_arms=valid_arms, explore=explore)
        action = self.bandit.arm_to_action(arm_idx)
        return action, context, scores

    # ---------------------------------------------------------------------------
    # State Representation & Discretization
    # ---------------------------------------------------------------------------

    def discretize_state(self, conv_state: ConversationState,
                         csp_queried: bool = False,
                         bayes_queried: bool = False) -> str:
        """
        Converts the dynamic conversation state into a discrete hashable key.
        """
        claim = conv_state.claim
        csp = conv_state.csp_result
        bayes = conv_state.risk_assessment

        # Amount tier: 0 (low <= $100), 1 (mid <= $300), 2 (high > $300)
        amt = claim.amount if claim else 0.0
        if amt <= 100:
            amt_tier = "L"
        elif amt <= 300:
            amt_tier = "M"
        else:
            amt_tier = "H"

        # Threat tier
        threat = claim.threat_level.value if claim else "none"

        # CSP status
        if not csp_queried or csp is None:
            csp_status = "CSP_UNCHECKED"
        else:
            n_sat = sum(1 for s in csp.constraints.values() if s == ConstraintStatus.SATISFIED)
            n_vio = sum(1 for s in csp.constraints.values() if s == ConstraintStatus.VIOLATED)
            n_unc = sum(1 for s in csp.constraints.values() if s == ConstraintStatus.UNCERTAIN)
            n_man = sum(1 for s in csp.constraints.values() if s == ConstraintStatus.POTENTIALLY_MANIPULATED)

            if n_vio > 0:
                csp_status = "CSP_VIOLATED"
            elif n_man > 0:
                csp_status = "CSP_MANIPULATED"
            elif n_unc > 0:
                csp_status = "CSP_UNCERTAIN"
            elif n_sat > 0:
                csp_status = "CSP_SATISFIED"
            else:
                csp_status = "CSP_EMPTY"

        # Bayes status
        if not bayes_queried or bayes is None:
            bayes_status = "BAYES_UNCHECKED"
        else:
            p = bayes.fraud_probability
            if p >= 0.65:
                bayes_status = "FRAUD_HIGH"
            elif p >= 0.35:
                bayes_status = "FRAUD_MED"
            else:
                bayes_status = "FRAUD_LOW"

        turn_tier = min(conv_state.turn, 4)
        verif_count = min(len(conv_state.completed_verifications), 3)

        evidence_verified = int("evidence_check" in conv_state.completed_verifications)
        history_verified = int("history_check" in conv_state.completed_verifications)
        order_verified = int("order_lookup" in conv_state.completed_verifications)
        return (
            f"A:{amt_tier}|T:{threat}|C:{csp_status}|B:{bayes_status}|"
            f"Trn:{turn_tier}|V:{verif_count}|E:{evidence_verified}|"
            f"H:{history_verified}|O:{order_verified}"
        )

    # ---------------------------------------------------------------------------
    # Action Selection with Action Masking
    # ---------------------------------------------------------------------------

    def get_valid_actions(self, conv_state: ConversationState,
                          csp_queried: bool,
                          bayes_queried: bool) -> List[ActionType]:
        """
        Action masking to prevent nonsensical loops:
        - Cannot call CSP twice in a row without state change
        - Cannot call Bayes before CSP has flagged UNCERTAIN or MANIPULATED
        - Cannot approve if violations are known or CSP has not run
        """
        valid = []

        # 1. Internal epistemic actions
        if not csp_queried:
            return [ActionType.CALL_CSP]

        if csp_queried and not bayes_queried:
            # Bayes is relevant if CSP found uncertain or manipulated flags
            if conv_state.csp_result and (conv_state.csp_result.has_uncertain or
                                          conv_state.csp_result.has_manipulated):
                return [ActionType.CALL_BAYES]

        # 2. External communicative actions
        # Can always ask for evidence or investigate if not fully satisfied
        valid.append(ActionType.ASK_FOR_EVIDENCE)
        valid.append(ActionType.INVESTIGATE)
        valid.append(ActionType.OFFER_ALTERNATIVE)
        valid.append(ActionType.ESCALATE)

        # Can reject if violation exists or high fraud detected
        if conv_state.csp_result and conv_state.csp_result.violated_policies:
            valid.append(ActionType.REJECT)
        elif conv_state.risk_assessment and conv_state.risk_assessment.fraud_probability > 0.60:
            valid.append(ActionType.REJECT)
        # Can only approve if CSP has passed without unresolved violations/manipulations
        if csp_queried and conv_state.csp_result:
            no_violations = len(conv_state.csp_result.violated_policies) == 0
            no_manipulations = not conv_state.csp_result.has_manipulated
            all_satisfied = all(
                status == ConstraintStatus.SATISFIED
                for status in conv_state.csp_result.constraints.values()
            )
            # Fraud must be low if evaluated
            fraud_safe = True
            if bayes_queried and conv_state.risk_assessment:
                fraud_safe = conv_state.risk_assessment.fraud_probability < 0.35

            claim = conv_state.claim
            verifications = set(conv_state.completed_verifications)
            required_verifications = {"order_lookup"}
            if claim and claim.intent != "cancellation":
                required_verifications.update({"evidence_check", "history_check"})
                if claim.amount > 200:
                    required_verifications.add("manager_approval")
            verified = required_verifications.issubset(verifications)

            if no_violations and no_manipulations and all_satisfied and fraud_safe and verified:
                valid.append(ActionType.APPROVE)

        if not valid:
            valid = [ActionType.ASK_FOR_EVIDENCE, ActionType.ESCALATE]

        return valid

    def select_action(self, conv_state: ConversationState,
                      csp_queried: bool,
                      bayes_queried: bool,
                      force_exploitation: bool = False) -> Tuple[ActionType, Dict[str, float]]:
        """
        Action selection over valid action mask.
        1. If epistemic action needed (CALL_CSP, CALL_BAYES), select it deterministically.
        2. For external macro actions, use the trained 54,000-state Tabular Q-Table if loaded.
        3. Fall back to discretized hash Q-table if 54k table is not available.
        """
        valid_actions = self.get_valid_actions(conv_state, csp_queried, bayes_queried)

        # 1. Handle epistemic actions
        if valid_actions in ([ActionType.CALL_CSP], [ActionType.CALL_BAYES]):
            chosen_action = valid_actions[0]
            return chosen_action, {chosen_action.value: 100.0}

        # 2. For external actions, utilize the 54,000-state Tabular Q-Table (Idea 1)
        if self.q_table_54k is not None and not any(
            a in valid_actions for a in (ActionType.CALL_CSP, ActionType.CALL_BAYES)
        ):
            chosen_action, state_id, q_dict = self.select_q_action(
                conv_state, valid_actions=valid_actions, force_exploitation=force_exploitation
            )
            return chosen_action, q_dict

        # 3. Fallback to discrete hash Q-table
        state_key = self.discretize_state(conv_state, csp_queried, bayes_queried)
        if state_key not in self.q_table:
            self.q_table[state_key] = {a.value: 0.0 for a in self.ALL_ACTIONS}
            self._seed_state_values(self.q_table[state_key], conv_state)

        q_vals = self.q_table[state_key]

        # Exploration vs Exploitation
        if not force_exploitation and random.random() < self.epsilon:
            chosen_action = random.choice(valid_actions)
        else:
            best_val = float('-inf')
            best_acts = []
            for act in valid_actions:
                val = q_vals.get(act.value, 0.0)
                if val > best_val:
                    best_val = val
                    best_acts = [act]
                elif val == best_val:
                    best_acts.append(act)
            chosen_action = best_acts[0]

        return chosen_action, q_vals

    def _seed_state_values(self, values: Dict[str, float],
                           conv_state: ConversationState) -> None:
        """Give unseen states safe, deterministic cold-start preferences."""
        csp = conv_state.csp_result
        risk = conv_state.risk_assessment
        if csp is None:
            values[ActionType.CALL_CSP.value] = 50.0
        elif csp.violated_policies:
            values[ActionType.REJECT.value] = 80.0
            values[ActionType.OFFER_ALTERNATIVE.value] = 35.0
        elif csp.has_manipulated or (
            risk is not None and risk.fraud_probability >= 0.65
        ):
            values[ActionType.ESCALATE.value] = 80.0
            values[ActionType.INVESTIGATE.value] = 45.0
        elif csp.has_uncertain:
            missing_evidence = any(
                status == ConstraintStatus.UNCERTAIN and
                any(term in name for term in ("receipt", "photo"))
                for name, status in csp.constraints.items()
            )
            values[ActionType.ASK_FOR_EVIDENCE.value] = 70.0 if missing_evidence else 30.0
            values[ActionType.INVESTIGATE.value] = 40.0
            values[ActionType.ESCALATE.value] = (
                65.0 if any(
                    status == ConstraintStatus.UNCERTAIN and "manager_approval" in name
                    for name, status in csp.constraints.items()
                ) and not missing_evidence else 20.0
            )
        elif all(s == ConstraintStatus.SATISFIED for s in csp.constraints.values()):
            claim = conv_state.claim
            required = {"order_lookup"}
            if claim and claim.intent != "cancellation":
                required.update({"evidence_check", "history_check"})
                if claim.amount > 200:
                    required.add("manager_approval")
            missing = required - set(conv_state.completed_verifications)
            if not missing:
                values[ActionType.APPROVE.value] = 70.0
            elif "manager_approval" in missing:
                values[ActionType.ESCALATE.value] = 70.0
            elif "evidence_check" in missing and not (
                claim and claim.evidence_provided
            ):
                values[ActionType.ASK_FOR_EVIDENCE.value] = 60.0
            else:
                values[ActionType.INVESTIGATE.value] = 45.0
        else:
            values[ActionType.INVESTIGATE.value] = 40.0

    # ---------------------------------------------------------------------------
    # Reward Calculation & Q-Value Update
    # ---------------------------------------------------------------------------

    def calculate_reward(self, conv_state: ConversationState,
                         action: ActionType,
                         csp_queried: bool,
                         bayes_queried: bool,
                         ground_truth_fraud: Optional[bool] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculates multi-objective scalar reward:
        + Correct resolution (+100 approval of genuine, +80 rejection of fraud)
        - Severe penalty for policy violations (-250)
        - Financial loss penalty proportional to amount (-amount * 0.5)
        - Unnecessary escalation penalty (-30)
        - False rejection penalty (-120)
        - Small step / investigation cost (-5)
        - Epistemic discovery reward (+15 for running CSP/Bayes when appropriate)
        """
        breakdown = {
            "resolution": 0.0,
            "compliance": 0.0,
            "financial": 0.0,
            "cost": -2.0,  # small step cost
            "discovery": 0.0,
        }

        claim = conv_state.claim
        csp = conv_state.csp_result
        bayes = conv_state.risk_assessment
        amt = claim.amount if claim else 0.0

        # Reward for epistemic actions
        if action == ActionType.CALL_CSP:
            breakdown["discovery"] = 15.0
            return sum(breakdown.values()), breakdown

        if action == ActionType.CALL_BAYES:
            if csp and (csp.has_uncertain or csp.has_manipulated):
                breakdown["discovery"] = 25.0
            else:
                breakdown["discovery"] = -10.0  # Called Bayes unnecessarily
            return sum(breakdown.values()), breakdown

        # External actions
        if action == ActionType.APPROVE:
            # Check compliance
            if csp and csp.violated_policies:
                breakdown["compliance"] = -250.0
                breakdown["financial"] = -amt
            elif bayes and bayes.fraud_probability > 0.60:
                breakdown["compliance"] = -200.0
                breakdown["financial"] = -amt
            elif ground_truth_fraud is True:
                breakdown["compliance"] = -250.0
                breakdown["financial"] = -amt
            else:
                # Legitimate approval
                breakdown["resolution"] = 120.0
                breakdown["compliance"] = 40.0

        elif action == ActionType.REJECT:
            if csp and csp.violated_policies:
                breakdown["resolution"] = 90.0
                breakdown["compliance"] = 40.0
            elif bayes and bayes.fraud_probability > 0.60:
                breakdown["resolution"] = 90.0
                breakdown["compliance"] = 30.0
            elif ground_truth_fraud is False:
                # False rejection of a genuine user
                breakdown["compliance"] = -120.0
            else:
                breakdown["resolution"] = 20.0

        elif action == ActionType.ASK_FOR_EVIDENCE:
            if csp and csp.has_uncertain:
                breakdown["resolution"] = 25.0
            else:
                breakdown["resolution"] = -10.0  # Evidence was already sufficient

        elif action == ActionType.INVESTIGATE:
            if bayes and bayes.fraud_probability > 0.35:
                breakdown["resolution"] = 20.0
            else:
                breakdown["resolution"] = 5.0

        elif action == ActionType.ESCALATE:
            # Appropriate if high value or complex manipulation
            if (amt > 200) or (csp and csp.has_manipulated) or (claim and claim.threat_level == ThreatLevel.HIGH):
                breakdown["resolution"] = 35.0
            else:
                breakdown["cost"] -= 30.0  # Unnecessary escalation

        elif action == ActionType.OFFER_ALTERNATIVE:
            if csp and csp.violated_policies:
                breakdown["resolution"] = 45.0  # Good customer de-escalation
            else:
                breakdown["resolution"] = 10.0

        total = sum(breakdown.values())
        return total, breakdown

    def update_q_value(self, state_key: str, action: ActionType,
                       reward: float, next_state_key: str, done: bool = False):
        """Standard Bellman equation update."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {a.value: 0.0 for a in self.ALL_ACTIONS}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a.value: 0.0 for a in self.ALL_ACTIONS}

        current_q = self.q_table[state_key][action.value]
        if done:
            target = reward
        else:
            max_next_q = max(self.q_table[next_state_key].values())
            target = reward + self.gamma * max_next_q

        self.q_table[state_key][action.value] = current_q + self.lr * (target - current_q)

    # ---------------------------------------------------------------------------
    # Baseline Policies Comparison
    # ---------------------------------------------------------------------------

    @staticmethod
    def baseline_always_escalate(conv_state: ConversationState, csp_queried: bool,
                                 bayes_queried: bool) -> ActionType:
        """Fixed heuristic 1: Always escalate uncertain cases."""
        if not csp_queried:
            return ActionType.CALL_CSP
        if conv_state.csp_result and (conv_state.csp_result.has_uncertain or conv_state.csp_result.has_manipulated):
            return ActionType.ESCALATE
        if conv_state.csp_result and conv_state.csp_result.violated_policies:
            return ActionType.REJECT
        return ActionType.APPROVE

    @staticmethod
    def baseline_naive_approve(conv_state: ConversationState, csp_queried: bool,
                               bayes_queried: bool) -> ActionType:
        """Fixed heuristic 2: Naively approve whenever any documents or demands exist."""
        if conv_state.claim and conv_state.claim.evidence_provided:
            return ActionType.APPROVE
        return ActionType.ASK_FOR_EVIDENCE

    @staticmethod
    def baseline_strict_reject(conv_state: ConversationState, csp_queried: bool,
                               bayes_queried: bool) -> ActionType:
        """Fixed heuristic 3: Reject on any uncertainty or threat."""
        if conv_state.claim and conv_state.claim.threat_level in (ThreatLevel.HIGH, ThreatLevel.MEDIUM):
            return ActionType.REJECT
        if not csp_queried:
            return ActionType.CALL_CSP
        if conv_state.csp_result and conv_state.csp_result.has_uncertain:
            return ActionType.REJECT
        return ActionType.APPROVE

    # ---------------------------------------------------------------------------
    # Persistence & Pre-training
    # ---------------------------------------------------------------------------

    def _load_or_init_q_table(self):
        """Loads saved Q-table or initializes with domain-grounded heuristics."""
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, "r", encoding="utf-8") as f:
                    self.q_table = json.load(f)
                    return
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not load Q-table %s: %s", self.q_table_path, exc)

        # Seed with high-reward priors reflecting domain policies
        self._seed_q_table_priors()

    def _seed_q_table_priors(self):
        """Seeds Q-table with sensible initial values so cold-start performs well."""
        # When CSP unchecked -> CALL_CSP should have high value
        # When CSP has uncertainty -> CALL_BAYES or ASK_FOR_EVIDENCE should be high
        # When Violations present -> REJECT or OFFER_ALTERNATIVE high
        # When Satisfied and Fraud low -> APPROVE high
        for amt in ["L", "M", "H"]:
            for threat in ["none", "low", "medium", "high"]:
                # State: CSP unchecked
                key1 = (
                    f"A:{amt}|T:{threat}|C:CSP_UNCHECKED|B:BAYES_UNCHECKED|"
                    "Trn:0|V:0|E:0|H:0|O:0"
                )
                self.q_table[key1] = {a.value: 0.0 for a in self.ALL_ACTIONS}
                self.q_table[key1][ActionType.CALL_CSP.value] = 50.0

                # State: CSP uncertain
                key2 = (
                    f"A:{amt}|T:{threat}|C:CSP_UNCERTAIN|B:BAYES_UNCHECKED|"
                    "Trn:1|V:0|E:0|H:0|O:0"
                )
                self.q_table[key2] = {a.value: 0.0 for a in self.ALL_ACTIONS}
                self.q_table[key2][ActionType.CALL_BAYES.value] = 45.0
                self.q_table[key2][ActionType.ASK_FOR_EVIDENCE.value] = 30.0

                # State: CSP violated
                key3 = (
                    f"A:{amt}|T:{threat}|C:CSP_VIOLATED|B:BAYES_UNCHECKED|"
                    "Trn:1|V:0|E:0|H:0|O:0"
                )
                self.q_table[key3] = {a.value: 0.0 for a in self.ALL_ACTIONS}
                self.q_table[key3][ActionType.REJECT.value] = 60.0
                self.q_table[key3][ActionType.OFFER_ALTERNATIVE.value] = 35.0

                # State: CSP satisfied, Bayes low
                key4 = (
                    f"A:{amt}|T:{threat}|C:CSP_SATISFIED|B:FRAUD_LOW|"
                    "Trn:2|V:1|E:0|H:0|O:1"
                )
                self.q_table[key4] = {a.value: 0.0 for a in self.ALL_ACTIONS}
                self.q_table[key4][ActionType.APPROVE.value] = 75.0

    def save_q_table(self):
        """Persist Q-table to disk."""
        try:
            with open(self.q_table_path, "w", encoding="utf-8") as f:
                json.dump(self.q_table, f, indent=2)
        except OSError as exc:
            logger.error("Could not save Q-table %s: %s", self.q_table_path, exc)
