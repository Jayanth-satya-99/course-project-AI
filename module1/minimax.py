# Module 1 — Minimax Search with Alpha-Beta Pruning
# ===================================================
# Models the adversarial interaction between a policy-defending service agent
# and a manipulative customer.
#
# The AGENT (maximiser) tries to maximise policy compliance.
# The USER (minimiser) tries to minimise it (obtain unauthorised outcomes).
#
# Alpha-Beta pruning eliminates branches that cannot affect the final decision,
# reducing the search space while guaranteeing the same optimal result.

from dataclasses import dataclass
from typing import List, Optional, Tuple

from shared.models import (
    ClaimExtraction, ActionType, MinimaxResult, ThreatLevel, Sentiment
)


@dataclass
class DialogueMove:
    """A single move in the dialogue game tree."""
    move_id: str
    description: str
    tone: str           # "firm_polite", "empathetic", "aggressive", "yielding"
    compliance: float   # how policy-compliant this move is [0, 1]
    empathy: float      # how empathetic/human this move sounds [0, 1]
    risk: float         # risk of policy violation if this move is taken [0, 1]

    @property
    def score(self) -> float:
        """Combined utility score for the agent (maximiser)."""
        return (self.compliance * 0.5) + (self.empathy * 0.3) - (self.risk * 0.2)


@dataclass
class GameNode:
    """A node in the Minimax game tree."""
    depth: int
    is_agent_turn: bool   # True = agent (maximiser), False = user (minimiser)
    move: Optional[DialogueMove]
    children: List['GameNode']
    value: Optional[float] = None


class MinimaxEngine:
    """
    Minimax search engine with Alpha-Beta pruning for tactical dialogue defence.

    Given Module 4's strategic goal (e.g., ASK_FOR_EVIDENCE) and the user's
    threat profile, this engine builds a depth-limited dialogue tree and
    selects the safest conversational phrasing that maintains policy compliance
    while resisting emotional pressure.
    """

    MAX_DEPTH = 4  # Depth-limited search to keep it tractable

    def __init__(self):
        self.nodes_evaluated = 0
        self.branches_pruned = 0

    def search(self, action: ActionType, claim: ClaimExtraction,
               policy_clauses: List[str]) -> MinimaxResult:
        """
        Run Minimax search with Alpha-Beta pruning.

        Args:
            action: The strategic goal from Module 4 (e.g., ASK_FOR_EVIDENCE).
            claim: The customer's extracted claim.
            policy_clauses: Relevant policy clause IDs.

        Returns:
            MinimaxResult with the optimal tactical response.
        """
        self.nodes_evaluated = 0
        self.branches_pruned = 0

        # Build the game tree
        root = self._build_game_tree(action, claim, depth=0)

        # Run alpha-beta search
        best_value = self._alpha_beta(root, self.MAX_DEPTH,
                                      float('-inf'), float('inf'),
                                      is_maximising=True)

        # Find the best first-level move
        best_move = self._select_best_child(root)

        return MinimaxResult(
            selected_response_type=best_move.move.tone if best_move.move else "firm_polite",
            compliance_score=best_move.value if best_move.value is not None else 0.5,
            pruned_branches=self.branches_pruned,
            evaluated_nodes=self.nodes_evaluated,
            reasoning=self._explain_selection(best_move, action, claim)
        )

    def _build_game_tree(self, action: ActionType, claim: ClaimExtraction,
                         depth: int) -> GameNode:
        """Recursively build the dialogue game tree."""
        if depth >= self.MAX_DEPTH:
            return GameNode(depth=depth, is_agent_turn=(depth % 2 == 0),
                            move=None, children=[])

        is_agent_turn = (depth % 2 == 0)

        if is_agent_turn:
            moves = self._generate_agent_moves(action, claim)
        else:
            moves = self._generate_user_moves(claim)

        children = []
        for move in moves:
            child = GameNode(
                depth=depth,
                is_agent_turn=is_agent_turn,
                move=move,
                children=[]
            )
            # Recursively build subtree
            child.children = [
                self._build_game_tree(action, claim, depth + 1)
            ] if depth + 1 < self.MAX_DEPTH else []
            children.append(child)

        return GameNode(depth=depth, is_agent_turn=is_agent_turn,
                        move=None, children=children)

    def _generate_agent_moves(self, action: ActionType,
                              claim: ClaimExtraction) -> List[DialogueMove]:
        """Generate possible dialogue moves for the agent based on the strategic goal."""
        moves = []

        if action == ActionType.ASK_FOR_EVIDENCE:
            moves = [
                DialogueMove("ag_firm_polite", "Politely and firmly request evidence",
                             "firm_polite", 0.95, 0.85, 0.05),
                DialogueMove("ag_empathetic", "Empathetically explain evidence need",
                             "empathetic", 0.90, 0.95, 0.10),
                DialogueMove("ag_aggressive", "Bluntly demand evidence",
                             "aggressive", 0.98, 0.30, 0.15),
                DialogueMove("ag_yielding", "Accept claim without full evidence",
                             "yielding", 0.20, 0.90, 0.85),
            ]
        elif action == ActionType.REJECT:
            moves = [
                DialogueMove("ag_firm_reject", "Firmly reject with policy citation",
                             "firm_polite", 0.95, 0.70, 0.05),
                DialogueMove("ag_empathetic_reject", "Empathetically reject with alternatives",
                             "empathetic", 0.90, 0.90, 0.08),
                DialogueMove("ag_harsh_reject", "Bluntly reject the claim",
                             "aggressive", 0.98, 0.20, 0.10),
                DialogueMove("ag_soft_reject", "Vaguely reject, leaving room for argument",
                             "yielding", 0.60, 0.85, 0.40),
            ]
        elif action == ActionType.APPROVE:
            moves = [
                DialogueMove("ag_approve_standard", "Approve with standard confirmation",
                             "firm_polite", 0.95, 0.80, 0.05),
                DialogueMove("ag_approve_warm", "Warmly approve and reassure",
                             "empathetic", 0.90, 0.95, 0.05),
            ]
        elif action == ActionType.INVESTIGATE:
            moves = [
                DialogueMove("ag_investigate_polite", "Politely explain investigation needed",
                             "firm_polite", 0.95, 0.80, 0.05),
                DialogueMove("ag_investigate_empathetic", "Empathetically describe the process",
                             "empathetic", 0.88, 0.92, 0.08),
                DialogueMove("ag_investigate_direct", "Directly state investigation is required",
                             "aggressive", 0.95, 0.40, 0.10),
            ]
        elif action == ActionType.ESCALATE:
            moves = [
                DialogueMove("ag_escalate_polite", "Politely offer escalation to manager",
                             "firm_polite", 0.95, 0.85, 0.05),
                DialogueMove("ag_escalate_empathetic", "Empathetically escalate the case",
                             "empathetic", 0.90, 0.95, 0.05),
            ]
        elif action == ActionType.OFFER_ALTERNATIVE:
            moves = [
                DialogueMove("ag_alt_firm", "Firmly offer alternative with policy basis",
                             "firm_polite", 0.90, 0.80, 0.08),
                DialogueMove("ag_alt_empathetic", "Empathetically suggest alternatives",
                             "empathetic", 0.85, 0.95, 0.10),
                DialogueMove("ag_alt_generous", "Offer generous alternative beyond policy",
                             "yielding", 0.50, 0.95, 0.45),
            ]
        else:
            # Default fallback
            moves = [
                DialogueMove("ag_default_firm", "Standard firm response",
                             "firm_polite", 0.85, 0.75, 0.10),
                DialogueMove("ag_default_empathetic", "Standard empathetic response",
                             "empathetic", 0.80, 0.90, 0.12),
            ]

        return moves

    def _generate_user_moves(self, claim: ClaimExtraction) -> List[DialogueMove]:
        """
        Generate possible adversarial user responses based on their threat profile.
        The user (minimiser) tries to lower the agent's compliance score.
        """
        base_moves = [
            DialogueMove("usr_comply", "Customer cooperates and provides info",
                         "cooperative", 0.0, 0.0, 0.0),  # Best for agent
            DialogueMove("usr_emotional", "Customer makes emotional appeal",
                         "emotional", 0.0, 0.0, 0.3),
            DialogueMove("usr_threaten", "Customer threatens chargeback or legal action",
                         "threatening", 0.0, 0.0, 0.6),
            DialogueMove("usr_escalate", "Customer demands to speak with manager",
                         "escalating", 0.0, 0.0, 0.4),
        ]

        # Adjust based on threat level
        if claim.threat_level == ThreatLevel.HIGH:
            base_moves.append(
                DialogueMove("usr_manipulate", "Customer fabricates evidence or lies",
                             "manipulative", 0.0, 0.0, 0.8)
            )
            base_moves.append(
                DialogueMove("usr_pressure", "Customer uses extreme pressure tactics",
                             "aggressive", 0.0, 0.0, 0.7)
            )

        # Score user moves: minimiser wants to minimise agent's score
        for move in base_moves:
            # User move score = negative of the risk it poses to the agent
            move.compliance = -move.risk
            move.empathy = 0.0

        return base_moves

    def _alpha_beta(self, node: GameNode, depth: int,
                    alpha: float, beta: float,
                    is_maximising: bool) -> float:
        """
        Alpha-Beta pruning implementation.

        Alpha = best value the maximiser can guarantee.
        Beta  = best value the minimiser can guarantee.
        Prune when alpha >= beta (no need to explore further).
        """
        self.nodes_evaluated += 1

        # Terminal node or depth limit reached
        if depth == 0 or not node.children:
            if node.move:
                return node.move.score
            return 0.5  # neutral default

        if is_maximising:
            max_eval = float('-inf')
            for child in node.children:
                if child.move:
                    child_val = child.move.score
                else:
                    child_val = 0.5

                # Recurse into child's subtree
                for grandchild in child.children:
                    sub_val = self._alpha_beta(grandchild, depth - 1,
                                              alpha, beta, False)
                    child_val = max(child_val, sub_val)

                child.value = child_val
                max_eval = max(max_eval, child_val)
                alpha = max(alpha, child_val)

                if beta <= alpha:
                    self.branches_pruned += 1
                    break  # Beta cutoff — prune

            node.value = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for child in node.children:
                if child.move:
                    child_val = child.move.score
                else:
                    child_val = 0.5

                for grandchild in child.children:
                    sub_val = self._alpha_beta(grandchild, depth - 1,
                                              alpha, beta, True)
                    child_val = min(child_val, sub_val)

                child.value = child_val
                min_eval = min(min_eval, child_val)
                beta = min(beta, child_val)

                if beta <= alpha:
                    self.branches_pruned += 1
                    break  # Alpha cutoff — prune

            node.value = min_eval
            return min_eval

    def _select_best_child(self, root: GameNode) -> GameNode:
        """Select the child with the highest value (best move for the agent)."""
        if not root.children:
            return root

        best = max(root.children,
                   key=lambda c: c.value if c.value is not None else float('-inf'))
        return best

    def _explain_selection(self, node: GameNode, action: ActionType,
                           claim: ClaimExtraction) -> str:
        """Generate a human-readable explanation of why this move was selected."""
        if not node.move:
            return "Default response selected."

        move = node.move
        parts = [
            f"Selected '{move.tone}' response strategy for goal '{action.value}'.",
            f"Compliance score: {move.compliance:.2f}, "
            f"Empathy score: {move.empathy:.2f}, "
            f"Risk: {move.risk:.2f}.",
        ]

        if claim.threat_level in (ThreatLevel.HIGH, ThreatLevel.MEDIUM):
            parts.append(
                f"Customer threat level is {claim.threat_level.value} — "
                f"firmness prioritised to resist manipulation."
            )

        if move.tone == "firm_polite":
            parts.append("Firm but polite tone balances compliance with customer experience.")
        elif move.tone == "empathetic":
            parts.append("Empathetic tone selected to de-escalate while maintaining policy.")

        parts.append(
            f"Search stats: {self.nodes_evaluated} nodes evaluated, "
            f"{self.branches_pruned} branches pruned by Alpha-Beta."
        )

        return " ".join(parts)
