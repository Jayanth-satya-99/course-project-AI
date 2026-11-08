# Module 4 — Context Vector Builder for Contextual Bandit
# ========================================================
# Compresses the multimodal outputs of:
#   1. Module 4 NLU (Claim amount, dates, evidence presence, adversarial flags)
#   2. Module 1 CSP (Policy encoding, constraints satisfied ratio, violations, uncertainty)
#   3. Module 3 Bayesian Network (Fraud probability, evidence quality one-hot)
#   4. System Trackers (Interaction turns, accumulated investigation cost)
# into an exact normalized 1D NumPy array of 20 dimensions.

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np

from shared.models import (
    ClaimExtraction, CSPResult, RiskAssessment, ConstraintStatus,
    EvidenceStrength, ThreatLevel, ConversationState
)

CONTEXT_DIMENSION = 20

FEATURE_NAMES = [
    # Policy Encoding (Mod 1)
    "policy_p_return",
    "policy_p_refund",
    "policy_p_exchange",
    # Constraint Evaluation (Mod 1)
    "constraints_satisfied_ratio",
    "has_unsatisfied_constraint",
    "has_uncertain_constraint",
    # Fraud & Evidence Risk (Mod 3)
    "estimated_fraud_probability",
    "evidence_is_missing",
    "evidence_is_weak",
    "evidence_is_strong",
    "evidence_is_contradictory",
    # Financial & Temporal (NLU)
    "normalized_amount",
    "normalized_days_since",
    "has_evidence_attached",
    # Adversarial Tactics (NLU)
    "adversarial_threat",
    "adversarial_emotion",
    "adversarial_contradiction",
    "adversarial_persuasion",
    # Cost Trackers (Internal)
    "normalized_turns",
    "normalized_investigation_cost",
]


def extract_nlu_perception_features(claim: Optional[ClaimExtraction]) -> Dict[str, Any]:
    """
    Isolate raw perception layer variables from ClaimExtraction:
      - claim_amount: Float
      - days_since_transaction: Integer
      - has_provided_evidence: Binary (0 or 1)
      - adversarial_flags: 4 distinct binary flags [threat, emotional_appeal, contradiction, persuasive_strategy]
    """
    if claim is None:
        return {
            "claim_amount": 0.0,
            "days_since_transaction": 0,
            "has_provided_evidence": 0,
            "adversarial_flags": {
                "threat": 0,
                "emotional_appeal": 0,
                "contradiction": 0,
                "persuasive_strategy": 0,
            }
        }

    # 1. Claim Amount
    amount = float(claim.amount) if claim.amount is not None else 0.0

    # 2. Days Since Transaction
    days_since = 0
    if claim.purchase_date:
        try:
            # Handle YYYY-MM-DD format
            p_date = datetime.strptime(claim.purchase_date.strip()[:10], "%Y-%m-%d")
            diff = (datetime.now() - p_date).days
            days_since = max(0, diff)
        except Exception:
            days_since = 0

    # 3. Evidence Presence
    has_evidence = 1 if (claim.evidence_provided and len(claim.evidence_provided) > 0) else 0

    # 4. Adversarial Flags
    tactics = set(claim.adversarial_tactics or [])
    has_threat = 1 if (
        claim.threat_level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH)
        or bool(getattr(claim, "threats", []))
        or bool(tactics & {"legal_threat", "chargeback_threat", "social_media_threat", "authority_pressure"})
    ) else 0

    has_emotion = 1 if (
        bool(getattr(claim, "emotional_appeals", []))
        or "emotional_appeal" in tactics
        or "sympathy_exploitation" in getattr(claim, "persuasive_strategies", [])
    ) else 0

    has_contradiction = 1 if (
        bool(getattr(claim, "contradictory_claims", []))
        or "contradiction" in tactics
        or "fact_shifting" in getattr(claim, "persuasive_strategies", [])
    ) else 0

    has_persuasion = 1 if (
        bool(getattr(claim, "persuasive_strategies", []))
        or bool(tactics & {"policy_gaslighting", "urgency_pressure", "social_proof_manipulation"})
    ) else 0

    return {
        "claim_amount": amount,
        "days_since_transaction": days_since,
        "has_provided_evidence": has_evidence,
        "adversarial_flags": {
            "threat": has_threat,
            "emotional_appeal": has_emotion,
            "contradiction": has_contradiction,
            "persuasive_strategy": has_persuasion,
        }
    }


def build_context_vector(
    claim: Optional[ClaimExtraction],
    csp_result: Optional[CSPResult],
    risk_assessment: Optional[RiskAssessment],
    conv_state: Optional[ConversationState] = None,
    interaction_turns: int = 1,
    investigation_cost: float = 0.0,
    max_amount_ceiling: float = 1500.0,
    max_policy_window_days: float = 30.0,
) -> np.ndarray:
    """
    Constructs a strictly normalized 1D numerical NumPy array of shape (20,)
    representing the state space S for the Contextual Bandit.
    """
    features = np.zeros(CONTEXT_DIMENSION, dtype=np.float64)

    # -------------------------------------------------------------------------
    # 1. Policy Encoding (From Module 1) — Indices [0:3]
    # -------------------------------------------------------------------------
    active_policies = set(csp_result.applicable_policies) if csp_result else set()
    if not active_policies and claim:
        # Fallback to claim intent if CSP has not run
        if claim.intent in ("return", "exchange"):
            active_policies.add("P_RETURN" if claim.intent == "return" else "P_EXCHANGE")
        else:
            active_policies.add("P_REFUND")

    if "P_RETURN" in active_policies:
        features[0] = 1.0
    if "P_REFUND" in active_policies:
        features[1] = 1.0
    if "P_EXCHANGE" in active_policies:
        features[2] = 1.0

    # -------------------------------------------------------------------------
    # 2. Constraint Evaluation (From Module 1) — Indices [3:6]
    # -------------------------------------------------------------------------
    if csp_result and csp_result.constraints:
        total = len(csp_result.constraints)
        sat = sum(1 for s in csp_result.constraints.values() if s == ConstraintStatus.SATISFIED)
        features[3] = float(sat) / float(max(total, 1))

        has_vio = (
            any(s == ConstraintStatus.VIOLATED for s in csp_result.constraints.values())
            or bool(csp_result.violated_policies)
        )
        features[4] = 1.0 if has_vio else 0.0

        has_unc = (
            any(s in (ConstraintStatus.UNCERTAIN, ConstraintStatus.POTENTIALLY_MANIPULATED)
                for s in csp_result.constraints.values())
            or csp_result.has_uncertain
            or csp_result.has_manipulated
        )
        features[5] = 1.0 if has_unc else 0.0
    else:
        # Baseline before CSP evaluation
        features[3] = 0.0
        features[4] = 0.0
        features[5] = 1.0

    # -------------------------------------------------------------------------
    # 3. Fraud Risk & Evidence Quality (From Module 3) — Indices [6:11]
    # -------------------------------------------------------------------------
    if risk_assessment:
        features[6] = float(np.clip(risk_assessment.fraud_probability, 0.0, 1.0))
        ev_strength = risk_assessment.evidence_strength

        # One-hot encoding [missing, weak, strong, contradictory]
        if ev_strength == EvidenceStrength.MISSING:
            features[7] = 1.0
        elif ev_strength == EvidenceStrength.WEAK:
            features[8] = 1.0
        elif ev_strength == EvidenceStrength.STRONG:
            features[9] = 1.0
        elif ev_strength in (EvidenceStrength.CONTRADICTORY, EvidenceStrength.FABRICATED):
            features[10] = 1.0
        else:
            # Moderate maps intermediate (0.5 weak, 0.5 strong)
            features[8] = 0.5
            features[9] = 0.5
    else:
        # Default unassessed state
        features[6] = 0.25  # Neutral baseline prior
        features[7] = 1.0   # Missing evidence by default

    # -------------------------------------------------------------------------
    # 4. Perception Layer Variables (From Module 4 NLU) — Indices [11:18]
    # -------------------------------------------------------------------------
    nlu_vars = extract_nlu_perception_features(claim)

    # Financial Value: normalized_amount ∈ [0.0, 1.0]
    features[11] = float(np.clip(nlu_vars["claim_amount"] / max_amount_ceiling, 0.0, 1.0))

    # Time Factor: normalized_days_since ∈ [0.0, 1.0]
    features[12] = float(np.clip(nlu_vars["days_since_transaction"] / max_policy_window_days, 0.0, 1.0))

    # Evidence Presence
    features[13] = float(nlu_vars["has_provided_evidence"])

    # Adversarial Flags: [threat, emotion, contradiction, persuasion]
    features[14] = float(nlu_vars["adversarial_flags"]["threat"])
    features[15] = float(nlu_vars["adversarial_flags"]["emotional_appeal"])
    features[16] = float(nlu_vars["adversarial_flags"]["contradiction"])
    features[17] = float(nlu_vars["adversarial_flags"]["persuasive_strategy"])

    # -------------------------------------------------------------------------
    # 5. Cost & Interaction Trackers (Internal State) — Indices [18:20]
    # -------------------------------------------------------------------------
    turns = conv_state.turn if conv_state else interaction_turns
    features[18] = float(np.clip(turns / 10.0, 0.0, 1.0))

    cost = getattr(conv_state, "investigation_cost", investigation_cost) if conv_state else investigation_cost
    features[19] = float(np.clip(cost / 50.0, 0.0, 1.0))

    return features

