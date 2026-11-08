# Shared Data Models
# ==================
# Defines all data structures used across modules for the Policy-Adherent Agent pipeline.

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConstraintStatus(Enum):
    """Status tags assigned by Module 1's CSP solver to each policy constraint."""
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNCERTAIN = "UNCERTAIN"
    POTENTIALLY_MANIPULATED = "POTENTIALLY_MANIPULATED"


class ActionType(Enum):
    """
    Action space for the RL-Centric Orchestrator (Module 4).
    Includes both *external communicative actions* and *internal epistemic actions*.
    """
    # External actions (affect the customer)
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ASK_FOR_EVIDENCE = "ASK_FOR_EVIDENCE"
    INVESTIGATE = "INVESTIGATE"
    ESCALATE = "ESCALATE"
    OFFER_ALTERNATIVE = "OFFER_ALTERNATIVE"
    # Internal epistemic actions (Module 4 queries other modules as tools)
    CALL_CSP = "CALL_CSP"
    CALL_BAYES = "CALL_BAYES"


class ThreatLevel(Enum):
    """Categorisation of the customer's threat intensity."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(Enum):
    """Detected emotional state of the customer."""
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    PLEADING = "pleading"
    THREATENING = "threatening"
    COOPERATIVE = "cooperative"


class EvidenceStrength(Enum):
    """Categorisation of submitted evidence quality."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    MISSING = "missing"
    CONTRADICTORY = "contradictory"
    FABRICATED = "fabricated"


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class ClaimExtraction:
    """Output of Module 4 NLU — structured representation of the customer's message."""
    intent: str                          # e.g. "refund", "exchange", "complaint"
    amount: float                        # claimed monetary value
    item: str                            # product mentioned
    purchase_date: Optional[str]         # date string if mentioned
    order_id: Optional[str]              # order identifier if mentioned
    evidence_provided: List[str]         # e.g. ["receipt", "photo"]
    threat_level: ThreatLevel
    sentiment: Sentiment
    adversarial_tactics: List[str]       # e.g. ["chargeback_threat", "emotional_appeal"]
    raw_text: str                        # original customer message
    claimed_previous_claims: Optional[int] = None
    contradictory_claims: List[Dict[str, Any]] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    emotional_appeals: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    persuasive_strategies: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize claim extraction to a JSON-compatible dictionary."""
        return {
            "intent": self.intent,
            "amount": self.amount,
            "item": self.item,
            "purchase_date": self.purchase_date,
            "order_id": self.order_id,
            "evidence_provided": self.evidence_provided,
            "threat_level": self.threat_level.value if isinstance(self.threat_level, ThreatLevel) else str(self.threat_level),
            "sentiment": self.sentiment.value if isinstance(self.sentiment, Sentiment) else str(self.sentiment),
            "adversarial_tactics": self.adversarial_tactics,
            "raw_text": self.raw_text,
            "claimed_previous_claims": self.claimed_previous_claims,
            "contradictory_claims": self.contradictory_claims,
            "supporting_evidence": self.supporting_evidence,
            "emotional_appeals": self.emotional_appeals,
            "threats": self.threats,
            "persuasive_strategies": self.persuasive_strategies,
            "conversation_history": self.conversation_history,
        }

    def to_file(self, file_path: Union[str, Path]) -> None:
        """Save this claim extraction to a JSON input file."""
        Path(file_path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimExtraction":
        """Construct ClaimExtraction from a dictionary (e.g. loaded from JSON)."""
        threat = data.get("threat_level", "none")
        if isinstance(threat, str):
            try:
                threat = ThreatLevel(threat.lower())
            except ValueError:
                threat = ThreatLevel.NONE

        sentiment = data.get("sentiment", "neutral")
        if isinstance(sentiment, str):
            try:
                sentiment = Sentiment(sentiment.lower())
            except ValueError:
                sentiment = Sentiment.NEUTRAL

        return cls(
            intent=str(data.get("intent", "refund")),
            amount=float(data.get("amount", 0.0)),
            item=str(data.get("item", "item")),
            purchase_date=data.get("purchase_date"),
            order_id=data.get("order_id"),
            evidence_provided=list(data.get("evidence_provided", [])),
            threat_level=threat,
            sentiment=sentiment,
            adversarial_tactics=list(data.get("adversarial_tactics", [])),
            raw_text=str(data.get("raw_text", "")),
            claimed_previous_claims=data.get("claimed_previous_claims"),
            contradictory_claims=list(data.get("contradictory_claims", [])),
            supporting_evidence=list(data.get("supporting_evidence", [])),
            emotional_appeals=list(data.get("emotional_appeals", [])),
            threats=list(data.get("threats", [])),
            persuasive_strategies=list(data.get("persuasive_strategies", [])),
            conversation_history=list(data.get("conversation_history", [])),
        )

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "ClaimExtraction":
        """Load ClaimExtraction directly from a JSON file."""
        content = Path(file_path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(content))


@dataclass
class CSPResult:
    """Output of Module 1 CSP — constraint evaluation results."""
    constraints: Dict[str, ConstraintStatus]   # constraint_name → status
    applicable_policies: List[str]             # matched policy clause IDs
    violated_policies: List[str]               # policies that are clearly broken
    uncertain_constraints: List[str]           # constraints needing Module 3
    has_uncertain: bool
    has_manipulated: bool
    explanation: str                           # human-readable summary
    evidence_graph: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CSPResult to a JSON-serializable dictionary."""
        return {
            "constraints": {
                k: (v.value if isinstance(v, ConstraintStatus) else str(v))
                for k, v in self.constraints.items()
            },
            "applicable_policies": self.applicable_policies,
            "violated_policies": self.violated_policies,
            "uncertain_constraints": self.uncertain_constraints,
            "has_uncertain": self.has_uncertain,
            "has_manipulated": self.has_manipulated,
            "explanation": self.explanation,
            "evidence_graph": self.evidence_graph,
        }

    def to_file(self, file_path: Union[str, Path]) -> None:
        """Save this CSP result to a JSON file."""
        Path(file_path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CSPResult":
        """Reconstruct a CSPResult from a dictionary."""
        constraints_raw = data.get("constraints", {})
        constraints = {}
        for k, v in constraints_raw.items():
            if isinstance(v, str):
                try:
                    constraints[k] = ConstraintStatus(v.lower())
                except ValueError:
                    constraints[k] = ConstraintStatus.UNCERTAIN
            elif isinstance(v, ConstraintStatus):
                constraints[k] = v
            else:
                constraints[k] = ConstraintStatus.UNCERTAIN

        return cls(
            constraints=constraints,
            applicable_policies=list(data.get("applicable_policies", [])),
            violated_policies=list(data.get("violated_policies", [])),
            uncertain_constraints=list(data.get("uncertain_constraints", [])),
            has_uncertain=bool(data.get("has_uncertain", False)),
            has_manipulated=bool(data.get("has_manipulated", False)),
            explanation=str(data.get("explanation", "")),
            evidence_graph=data.get("evidence_graph", {}),
        )

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "CSPResult":
        """Load a CSPResult directly from a JSON file."""
        content = Path(file_path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(content))


@dataclass
class RiskAssessment:
    """Output of Module 3 Bayesian Network — probabilistic risk evaluation."""
    fraud_probability: float           # P(fraud | evidence) ∈ [0, 1]
    evidence_reliability: float        # overall evidence quality score ∈ [0, 1]
    evidence_strength: EvidenceStrength
    risk_factors: List[str]            # list of contributing risk signals
    recommendation: str                # brief textual recommendation
    posterior_details: Dict[str, float]  # detailed posterior probabilities

    def to_dict(self) -> Dict[str, Any]:
        """Convert RiskAssessment to a JSON-serializable dictionary."""
        return {
            "fraud_probability": self.fraud_probability,
            "evidence_reliability": self.evidence_reliability,
            "evidence_strength": (
                self.evidence_strength.value
                if isinstance(self.evidence_strength, EvidenceStrength)
                else str(self.evidence_strength)
            ),
            "risk_factors": self.risk_factors,
            "recommendation": self.recommendation,
            "posterior_details": self.posterior_details,
        }

    def to_file(self, file_path: Union[str, Path]) -> None:
        """Save this risk assessment to a JSON file."""
        Path(file_path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskAssessment":
        """Reconstruct a RiskAssessment from a dictionary."""
        ev_strength = data.get("evidence_strength", "moderate")
        if isinstance(ev_strength, str):
            try:
                ev_strength = EvidenceStrength(ev_strength.lower())
            except ValueError:
                ev_strength = EvidenceStrength.MODERATE

        return cls(
            fraud_probability=float(data.get("fraud_probability", 0.0)),
            evidence_reliability=float(data.get("evidence_reliability", 0.0)),
            evidence_strength=ev_strength,
            risk_factors=list(data.get("risk_factors", [])),
            recommendation=str(data.get("recommendation", "")),
            posterior_details=dict(data.get("posterior_details", {})),
        )

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "RiskAssessment":
        """Load a RiskAssessment directly from a JSON file."""
        content = Path(file_path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(content))


@dataclass
class PlanStep:
    """A single step in Module 2's sequential resolution plan."""
    action: str                    # action identifier
    description: str               # human-readable description
    preconditions: List[str]       # what must be true before execution
    effects: List[str]             # what becomes true after execution
    completed: bool = False
    blocked: bool = False
    block_reason: str = ""


@dataclass
class PlanResult:
    """Output of Module 2 — the full resolution plan and execution status."""
    steps: List[PlanStep]
    current_step_index: int
    action_approved: bool          # whether Module 4's requested action is safe to execute
    block_reason: str              # why the action was blocked (if applicable)
    completed_steps: List[str]
    remaining_steps: List[str]


@dataclass
class MinimaxResult:
    """Output of Module 1 Minimax — the tactically optimal dialogue move."""
    selected_response_type: str    # e.g. "firm_polite", "empathetic_redirect"
    compliance_score: float        # how policy-compliant this move is
    pruned_branches: int           # branches eliminated by alpha-beta
    evaluated_nodes: int           # total nodes evaluated
    reasoning: str                 # why this move was selected


@dataclass
class GeneratedResponse:
    """Output of Module 5 GenAI — the final customer-facing response."""
    text: str                      # the actual response message
    cited_policies: List[str]      # policy clauses referenced
    evidence_status: str           # summary of evidence evaluation
    available_actions: List[str]   # what the customer can do next
    tone: str                      # detected tone of the response


@dataclass
class ConversationState:
    """
    Master state object that flows through the entire pipeline.
    Maintained across conversation turns.
    """
    session_id: str
    turn: int = 0
    claim: Optional[ClaimExtraction] = None
    csp_result: Optional[CSPResult] = None
    risk_assessment: Optional[RiskAssessment] = None
    plan_result: Optional[PlanResult] = None
    minimax_result: Optional[MinimaxResult] = None
    current_action: Optional[ActionType] = None
    resolved: bool = False
    resolution: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    completed_verifications: List[str] = field(default_factory=list)
    # Accumulated state for RL
    total_reward: float = 0.0
    policy_violations: int = 0
    false_rejections: int = 0

    def to_state_vector(self) -> Dict[str, float]:
        """Convert the conversation state into a numerical feature vector for the RL agent."""
        vector = {
            "turn": float(self.turn),
            "has_claim": 1.0 if self.claim else 0.0,
            "amount": self.claim.amount if self.claim else 0.0,
            "threat_level": {"none": 0.0, "low": 0.25, "medium": 0.5, "high": 1.0}.get(
                self.claim.threat_level.value, 0.0) if self.claim else 0.0,
            "num_evidence": float(len(self.claim.evidence_provided)) if self.claim else 0.0,
            "num_satisfied": 0.0,
            "num_violated": 0.0,
            "num_uncertain": 0.0,
            "num_manipulated": 0.0,
            "fraud_prob": self.risk_assessment.fraud_probability if self.risk_assessment else 0.5,
            "evidence_reliability": self.risk_assessment.evidence_reliability if self.risk_assessment else 0.5,
            "num_completed_verifications": float(len(self.completed_verifications)),
            "num_turns": float(self.turn),
        }
        if self.csp_result:
            for status in self.csp_result.constraints.values():
                if status == ConstraintStatus.SATISFIED:
                    vector["num_satisfied"] += 1.0
                elif status == ConstraintStatus.VIOLATED:
                    vector["num_violated"] += 1.0
                elif status == ConstraintStatus.UNCERTAIN:
                    vector["num_uncertain"] += 1.0
                elif status == ConstraintStatus.POTENTIALLY_MANIPULATED:
                    vector["num_manipulated"] += 1.0
        return vector
