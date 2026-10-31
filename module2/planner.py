# Module 2 — Sequential Planning (The Safety Sequencer)
# ======================================================
# Models claim resolution as a formal sequential planning problem with PDDL-style
# preconditions and effects.
#
# Primary Duties:
#   1. Generate an optimal sequence of verification steps to reach a resolution.
#   2. Act as a safety gatekeeper: Block irreversible actions (such as 'issue_refund')
#      until ALL necessary verification preconditions have been satisfied.
#   3. Dynamic replanning: Adapt the plan when new evidence is provided,
#      verifications fail, or manager escalation is triggered.

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple

from shared.models import (
    ActionType, ClaimExtraction, CSPResult, PlanResult, PlanStep,
    RiskAssessment, ConstraintStatus, ConversationState
)


@dataclass
class PlanningAction:
    """A formal planning operator with preconditions and effects."""
    name: str
    description: str
    preconditions: Set[str]
    effects: Set[str]
    is_irreversible: bool = False
    cost: float = 1.0


class SequentialPlanner:
    """
    PDDL-style forward search planner and safety sequencer.
    Defines the standard library of claim resolution actions.
    """

    def __init__(self):
        self.actions: Dict[str, PlanningAction] = self._build_action_library()

    def _build_action_library(self) -> Dict[str, PlanningAction]:
        """Defines all domain operators specified in the project requirements."""
        acts = [
            PlanningAction(
                name="retrieve_transaction_record",
                description="Query database for customer order & payment details",
                preconditions={"order_id_known"},
                effects={"transaction_retrieved", "purchase_date_verified", "amount_verified"},
                is_irreversible=False,
                cost=1.0
            ),
            PlanningAction(
                name="check_applicable_policy",
                description="Map request to policy clauses via policy graph",
                preconditions={"intent_classified"},
                effects={"policy_identified"},
                is_irreversible=False,
                cost=1.0
            ),
            PlanningAction(
                name="request_receipt",
                description="Prompt customer to submit purchase receipt or invoice",
                preconditions=set(),
                effects={"receipt_requested"},
                is_irreversible=False,
                cost=2.0
            ),
            PlanningAction(
                name="request_photo_evidence",
                description="Prompt customer for item defect/damage photo",
                preconditions=set(),
                effects={"photo_requested"},
                is_irreversible=False,
                cost=2.0
            ),
            PlanningAction(
                name="validate_submitted_evidence",
                description="Cross-check submitted documents/photos for authenticity",
                preconditions={"evidence_available"},
                effects={"evidence_validated"},
                is_irreversible=False,
                cost=3.0
            ),
            PlanningAction(
                name="inspect_previous_claims",
                description="Check historical frequency and return velocity of customer account",
                preconditions={"transaction_retrieved"},
                effects={"claim_history_checked"},
                is_irreversible=False,
                cost=1.5
            ),
            PlanningAction(
                name="contact_external_verification",
                description="Contact carrier/warehouse for delivery status confirmation",
                preconditions={"transaction_retrieved"},
                effects={"carrier_status_confirmed"},
                is_irreversible=False,
                cost=4.0
            ),
            PlanningAction(
                name="request_managerial_approval",
                description="Escalate high-value claim to human supervisor",
                preconditions={"evidence_validated", "claim_history_checked"},
                effects={"manager_approval_obtained"},
                is_irreversible=False,
                cost=5.0
            ),
            PlanningAction(
                name="issue_refund",
                description="Execute irreversible monetary disbursement to customer",
                preconditions={
                    "transaction_retrieved",
                    "policy_identified",
                    "evidence_validated",
                    "claim_history_checked",
                    "manager_approval_obtained",
                },
                effects={"refund_issued", "case_resolved"},
                is_irreversible=True,
                cost=10.0
            ),
            PlanningAction(
                name="complete_cancellation",
                description="Cancel an eligible order before shipment",
                preconditions={"transaction_retrieved", "policy_identified"},
                effects={"cancellation_completed", "case_resolved"},
                is_irreversible=True,
                cost=4.0
            ),
            PlanningAction(
                name="reject_request",
                description="Formally deny the customer's request under policy clause",
                preconditions={"policy_identified", "violation_confirmed"},
                effects={"request_rejected", "case_resolved"},
                is_irreversible=True,
                cost=5.0
            ),
            PlanningAction(
                name="offer_alternative_resolution",
                description="Offer store credit, exchange, or discounted repair",
                preconditions={"policy_identified"},
                effects={"alternative_offered", "case_resolved"},
                is_irreversible=False,
                cost=3.0
            ),
        ]
        return {a.name: a for a in acts}

    # ---------------------------------------------------------------------------
    # State Extraction from Domain Context
    # ---------------------------------------------------------------------------

    def extract_current_state(self, conv_state: ConversationState) -> Set[str]:
        """Derives current logical propositions from the master conversation state."""
        props = set()

        claim = conv_state.claim
        csp = conv_state.csp_result
        bayes = conv_state.risk_assessment

        if claim:
            props.add("intent_classified")
            if claim.amount <= 200 or claim.intent == "cancellation":
                props.add("manager_approval_obtained")
            if claim.order_id:
                props.add("order_id_known")
            if claim.evidence_provided:
                props.add("evidence_available")

        if csp:
            props.add("policy_identified")
            if not csp.violated_policies:
                props.add("no_blatant_violations")
            else:
                props.add("violation_confirmed")

        # Prior completed verifications
        for v in conv_state.completed_verifications:
            props.add(v)
            if v == "order_lookup":
                props.add("transaction_retrieved")
                props.add("order_id_known")
            elif v == "evidence_check":
                props.add("evidence_validated")
            elif v == "history_check":
                props.add("claim_history_checked")
            elif v == "manager_approval":
                props.add("manager_approval_obtained")
            elif v == "item_returned":
                props.add("return_confirmed")

        if bayes:
            if bayes.fraud_probability < 0.25 and bayes.evidence_reliability > 0.60:
                props.add("risk_acceptable")
            elif bayes.fraud_probability > 0.65:
                props.add("violation_confirmed")

        return props

    # ---------------------------------------------------------------------------
    # Safety Gatekeeper Verification
    # ---------------------------------------------------------------------------

    def verify_action_safety(self, requested_action: ActionType,
                             conv_state: ConversationState) -> Tuple[bool, str, List[str]]:
        """
        Validates whether the requested action is safe to execute right now.
        Returns:
            (is_approved, block_reason, missing_preconditions)
        """
        current_state = self.extract_current_state(conv_state)

        # Mapping external RL actions to planner operations
        action_map = {
            ActionType.APPROVE: "issue_refund",
            ActionType.REJECT: "reject_request",
            ActionType.OFFER_ALTERNATIVE: "offer_alternative_resolution",
            ActionType.ESCALATE: "request_managerial_approval",
        }

        op_name = action_map.get(requested_action)
        if not op_name:
            # Internal or non-terminal communicative actions (ASK_EVIDENCE, INVESTIGATE)
            return True, "", []

        if requested_action == ActionType.APPROVE:
            if not conv_state.csp_result:
                return False, "Policy constraints have not been evaluated.", ["policy_evaluation"]
            unresolved = [
                name for name, status in conv_state.csp_result.constraints.items()
                if status != ConstraintStatus.SATISFIED
            ]
            if unresolved:
                return False, (
                    "Approval is blocked until every applicable policy constraint "
                    "is satisfied: " + ", ".join(unresolved) + "."
                ), unresolved

            if claim := conv_state.claim:
                required = {"order_lookup"}
                if claim.intent != "cancellation":
                    required.update({"evidence_check", "history_check"})
                if claim.amount > 200 and claim.intent != "cancellation":
                    required.add("manager_approval")
                missing = sorted(required - set(conv_state.completed_verifications))
                if missing:
                    readable = ", ".join(item.replace("_", " ") for item in missing)
                    return False, f"Approval requires completed verification: {readable}.", missing
            return True, "", []

        operator = self.actions[op_name]

        # Irreversible Action Safety Gate: Preconditions MUST hold
        missing = [p for p in operator.preconditions if p not in current_state]

        if not missing:
            return True, "", []

        readable_missing = [m.replace("_", " ") for m in missing]
        reason = (f"Safety Gate blocked irreversible action '{op_name}'. "
                  f"Unsatisfied preconditions: {', '.join(readable_missing)}.")
        return False, reason, missing

    # ---------------------------------------------------------------------------
    # Forward Search Plan Generation
    # ---------------------------------------------------------------------------

    def generate_plan(self, target_goal: str,
                      conv_state: ConversationState) -> PlanResult:
        """
        Breadth-first search / forward state-space planning from current state
        to achieve target_goal (e.g., 'case_resolved', 'refund_issued', 'evidence_validated').
        """
        initial_state = self.extract_current_state(conv_state)
        goal_propositions = {target_goal}

        # Queue contains (current_props_set, action_list)
        from collections import deque
        queue = deque([(set(initial_state), [])])
        visited_states = []

        found_path: Optional[List[PlanningAction]] = None

        while queue:
            state, path = queue.popleft()

            # Check if all goal propositions are satisfied
            if goal_propositions.issubset(state):
                found_path = path
                break

            # State hashing for loop avoidance
            state_frozen = frozenset(state)
            if state_frozen in visited_states:
                continue
            visited_states.append(state_frozen)

            # Prevent overly long plans
            if len(path) >= 8:
                continue

            for act in self.actions.values():
                # Check preconditions
                if act.preconditions.issubset(state):
                    new_state = set(state) | act.effects
                    queue.append((new_state, path + [act]))

        steps: List[PlanStep] = []
        if found_path:
            for act in found_path:
                steps.append(PlanStep(
                    action=act.name,
                    description=act.description,
                    preconditions=list(act.preconditions),
                    effects=list(act.effects),
                    completed=False
                ))
        else:
            # Fallback heuristic steps if direct graph path is unreachable without user input
            steps = self._construct_heuristic_fallback_plan(initial_state, target_goal)

        # Mark completed steps based on initial_state
        completed_step_names = []
        remaining_step_names = []
        for s in steps:
            if set(s.effects).issubset(initial_state):
                s.completed = True
                completed_step_names.append(s.action)
            else:
                remaining_step_names.append(s.action)

        current_idx = 0
        for i, s in enumerate(steps):
            if not s.completed:
                current_idx = i
                break

        # Check safety of Module 4's chosen action
        act_approved, block_reason, _ = self.verify_action_safety(
            conv_state.current_action or ActionType.ASK_FOR_EVIDENCE, conv_state
        )

        return PlanResult(
            steps=steps,
            current_step_index=current_idx,
            action_approved=act_approved,
            block_reason=block_reason,
            completed_steps=completed_step_names,
            remaining_steps=remaining_step_names
        )

    def _construct_heuristic_fallback_plan(self, current_props: Set[str],
                                           target_goal: str) -> List[PlanStep]:
        """Provides an ordered workflow plan when information from customer is pending."""
        plan = []
        if "order_id_known" not in current_props:
            plan.append(PlanStep(
                action="request_receipt",
                description="Request proof of purchase or order ID from user",
                preconditions=[],
                effects=["receipt_requested", "evidence_available"]
            ))
        if "transaction_retrieved" not in current_props:
            plan.append(PlanStep(
                action="retrieve_transaction_record",
                description="Retrieve order record from customer database",
                preconditions=["order_id_known"],
                effects=["transaction_retrieved"]
            ))
        if "evidence_validated" not in current_props:
            plan.append(PlanStep(
                action="validate_submitted_evidence",
                description="Inspect photo or receipt authenticity",
                preconditions=["evidence_available"],
                effects=["evidence_validated"]
            ))
        if "claim_history_checked" not in current_props:
            plan.append(PlanStep(
                action="inspect_previous_claims",
                description="Query account return history for refund velocity",
                preconditions=["transaction_retrieved"],
                effects=["claim_history_checked"]
            ))
        if "manager_approval_obtained" not in current_props:
            plan.append(PlanStep(
                action="request_managerial_approval",
                description="Request human approval for a high-value claim",
                preconditions=["evidence_validated", "claim_history_checked"],
                effects=["manager_approval_obtained"]
            ))
        if "policy_identified" not in current_props:
            plan.append(PlanStep(
                action="check_applicable_policy",
                description="Evaluate the request against the applicable policy",
                preconditions=["intent_classified"],
                effects=["policy_identified"]
            ))
        plan.append(PlanStep(
            action="issue_refund",
            description="Process disbursement only after policy and verification checks",
            preconditions=[
                "transaction_retrieved", "policy_identified",
                "evidence_validated", "claim_history_checked"
            ],
            effects=["refund_issued", "case_resolved"]
        ))
        return plan
