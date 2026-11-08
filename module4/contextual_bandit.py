# Module 4 — LinUCB Contextual Bandit Engine
# ============================================
# Implements the Disjoint Linear Upper Confidence Bound (LinUCB) algorithm
# for the 6 strategic customer support decision arms.
#
# Each arm maintains independent parameters:
#   A_a = D_a^T D_a + I_d  (Covariance matrix ∈ ℝ²⁰ˣ²⁰)
#   b_a = D_a^T c_a        (Response vector ∈ ℝ²⁰)
#   θ̂_a = A_a⁻¹ b_a        (Ridge regression ridge weights)
#
# Upper Confidence Bound score:
#   p_a(x) = θ̂_a^T x + α √(x^T A_a⁻¹ x)

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from shared.models import ActionType

# 6 Discrete Bandit Arms
BANDIT_ARMS: List[ActionType] = [
    ActionType.APPROVE,            # Arm 0: Immediately grant claim
    ActionType.REJECT,             # Arm 1: Immediately deny claim
    ActionType.ASK_FOR_EVIDENCE,   # Arm 2: Prompt user for missing evidence
    ActionType.INVESTIGATE,        # Arm 3: Trigger internal API / order lookup check
    ActionType.OFFER_ALTERNATIVE,  # Arm 4: Offer store credit / replacement
    ActionType.ESCALATE,           # Arm 5: Transfer to senior human manager
]

ARM_NAMES: List[str] = [
    "APPROVE",
    "REJECT",
    "ASK_EVIDENCE",
    "PERFORM_VERIFICATION",
    "PROPOSE_ALTERNATIVE",
    "ESCALATE",
]


class LinUCBBandit:
    """
    Disjoint LinUCB Contextual Bandit with Sherman-Morrison rank-1 inverse updates
    for high-speed training and real-time decision inference.
    """

    def __init__(
        self,
        n_features: int = 20,
        n_arms: int = 6,
        alpha: float = 0.50,
        ridge_lambda: float = 1.0,
    ):
        self.n_features = n_features
        self.n_arms = n_arms
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)

        # Covariance matrices A_a and inverses A_inv_a
        self.A = [self.ridge_lambda * np.eye(n_features, dtype=np.float64) for _ in range(n_arms)]
        self.A_inv = [(1.0 / self.ridge_lambda) * np.eye(n_features, dtype=np.float64) for _ in range(n_arms)]

        # Response vectors b_a
        self.b = [np.zeros(n_features, dtype=np.float64) for _ in range(n_arms)]

        # Pull counters and rewards per arm
        self.arm_counts = np.zeros(n_arms, dtype=np.int64)
        self.arm_rewards = np.zeros(n_arms, dtype=np.float64)

    def select_arm(
        self,
        context: np.ndarray,
        valid_arms: Optional[List[int]] = None,
        explore: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Calculates UCB scores across arms and selects the arm with the highest value.
        Supports valid action masking.
        """
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if len(x) != self.n_features:
            raise ValueError(f"Context vector shape mismatch: expected {self.n_features}, got {len(x)}")

        scores = np.full(self.n_arms, -1e9, dtype=np.float64)
        candidates = valid_arms if valid_arms is not None else list(range(self.n_arms))
        if not candidates:
            candidates = list(range(self.n_arms))

        for a in candidates:
            A_inv_a = self.A_inv[a]
            b_a = self.b[a]

            # Ridge regression point estimate θ̂_a = A_a⁻¹ b_a
            theta_a = A_inv_a @ b_a
            expected_payoff = float(theta_a @ x)

            if explore and self.alpha > 0.0:
                # Variance / uncertainty metric: √(x^T A_a⁻¹ x)
                variance = float(x @ A_inv_a @ x)
                confidence_bound = self.alpha * np.sqrt(max(0.0, variance))
                scores[a] = expected_payoff + confidence_bound
            else:
                scores[a] = expected_payoff

        # Select arm maximizing UCB score among candidates
        best_arm = int(candidates[0])
        best_score = scores[best_arm]
        for a in candidates[1:]:
            if scores[a] > best_score:
                best_score = scores[a]
                best_arm = int(a)

        return best_arm, scores

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Online Bayesian linear regression update:
          A_a ← A_a + x x^T
          b_a ← b_a + r x
        Uses the Sherman-Morrison formula for exact, fast O(d²) inverse updates.
        """
        a = int(arm)
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        r = float(reward)

        # 1. Update covariance matrix and response vector
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
        self.arm_counts[a] += 1
        self.arm_rewards[a] += r

        # 2. Update inverse via Sherman-Morrison:
        # (A + x x^T)⁻¹ = A⁻¹ - (A⁻¹ x x^T A⁻¹) / (1 + x^T A⁻¹ x)
        A_inv = self.A_inv[a]
        A_inv_x = A_inv @ x
        denom = 1.0 + float(x @ A_inv_x)
        if denom > 1e-12:
            self.A_inv[a] -= np.outer(A_inv_x, A_inv_x) / denom
        else:
            # Fallback to direct numerical inversion if degenerate
            self.A_inv[a] = np.linalg.pinv(self.A[a])

    def arm_to_action(self, arm: int) -> ActionType:
        """Map discrete integer arm index to ActionType enum."""
        return BANDIT_ARMS[int(arm)]

    def action_to_arm(self, action: ActionType) -> int:
        """Map ActionType enum to discrete integer arm index."""
        if action in BANDIT_ARMS:
            return BANDIT_ARMS.index(action)
        # Fallbacks for epistemic / internal actions
        return 2  # ASK_FOR_EVIDENCE

    def save_weights(self, filepath: Union[str, Path]):
        """Persist learned bandit matrices to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            f"A_{i}": self.A[i] for i in range(self.n_arms)
        }
        data.update({f"b_{i}": self.b[i] for i in range(self.n_arms)})
        data["arm_counts"] = self.arm_counts
        data["arm_rewards"] = self.arm_rewards
        data["meta"] = np.array([self.n_features, self.n_arms, self.alpha, self.ridge_lambda])
        np.savez_compressed(str(path), **data)

    def load_weights(self, filepath: Union[str, Path]):
        """Load learned bandit matrices from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Bandit weights file not found: {filepath}")
        data = np.load(str(path))
        meta = data["meta"]
        self.n_features = int(meta[0])
        self.n_arms = int(meta[1])
        self.alpha = float(meta[2])
        self.ridge_lambda = float(meta[3])

        self.A = [data[f"A_{i}"] for i in range(self.n_arms)]
        self.A_inv = [np.linalg.pinv(self.A[i]) for i in range(self.n_arms)]
        self.b = [data[f"b_{i}"] for i in range(self.n_arms)]
        self.arm_counts = data["arm_counts"]
        self.arm_rewards = data["arm_rewards"]

