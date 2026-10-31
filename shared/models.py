# Shared Data Models
# ==================
# Defines all data structures used across modules for the Policy-Adherent Agent pipeline.

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


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


@dataclass
class RiskAssessment:
    """Output of Module 3 Bayesian Network — probabilistic risk evaluation."""
    fraud_probability: float           # P(fraud | evidence) ∈ [0, 1]
    evidence_reliability: float        # overall evidence quality score ∈ [0, 1]
    evidence_strength: EvidenceStrength
    risk_factors: List[str]            # list of contributing risk signals
    recommendation: str                # brief textual recommendation
    posterior_details: Dict[str, float]  # detailed posterior probabilities


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
