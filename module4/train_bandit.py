# Module 4 — Contextual Bandit Training & Evaluation Pipeline
# ============================================================
# Executes:
#   Phase A & B: 50,000+ Episode LinUCB Training in Custom Simulation Environment
#   Phase C: Control Baselines ("The Escaler" and "The Accommodator")
#   Phase D: Comparative Evaluation on 1,000 Unseen Cases

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from module4.context_vector import CONTEXT_DIMENSION
from module4.contextual_bandit import LinUCBBandit
from module4.environment import CustomerSupportBanditEnv


class Baseline1Escaler:
    """Baseline 1 (The Escaler): Risk-averse, highly inefficient."""
    name = "Baseline 1 (The Escaler)"

    def select_arm(self, context: np.ndarray) -> int:
        fraud_prob = float(context[6])
        # if fraud_prob > 0.35: return ESCALATE (Arm 5) else: return APPROVE (Arm 0)
        return 5 if fraud_prob > 0.35 else 0


class Baseline2Accommodator:
    """Baseline 2 (The Accommodator): Naive automated system."""
    name = "Baseline 2 (The Accommodator)"

    def select_arm(self, context: np.ndarray) -> int:
        has_evidence = float(context[13])
        turns = float(context[18])
        # if has_evidence_attached == 1: return APPROVE (Arm 0)
        if has_evidence >= 1.0:
            return 0
        # If already asked for evidence and customer still has none, naive accommodator approves anyway
        if turns >= 0.2:
            return 0
        # Otherwise: return ASK_EVIDENCE (Arm 2)
        return 2


def train_linucb_bandit(
    episodes: int = 50000,
    alpha: float = 0.50,
    weights_path: Optional[Union[str, Path]] = None,
    log_interval: Optional[int] = None,
) -> LinUCBBandit:
    """
    Trains LinUCB over simulated customer support episodes.
    """
    env = CustomerSupportBanditEnv(seed=42)
    bandit = LinUCBBandit(n_features=CONTEXT_DIMENSION, n_arms=6, alpha=alpha)

    print(f"\n=======================================================")
    print(f"Starting LinUCB Contextual Bandit Training: {episodes:,} episodes")
    print(f"=======================================================")
    start_time = time.time()

    interval = log_interval or max(100, min(10000, episodes // 5))
    cumulative_reward = 0.0
    recent_rewards: List[float] = []

    for ep in range(1, episodes + 1):
        context, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            # UCB Action Selection
            arm, _ = bandit.select_arm(context, explore=True)
            next_context, reward, done, _, _ = env.step(arm)

            # Online weight update
            bandit.update(arm, context, reward)

            context = next_context
            ep_reward += reward

        cumulative_reward += ep_reward
        recent_rewards.append(ep_reward)

        if ep % interval == 0 or ep == episodes:
            avg_recent = np.mean(recent_rewards[-interval:])
            print(
                f"Episode {ep:6d}/{episodes} | "
                f"Avg Reward (last {interval}): {avg_recent:+6.2f} | "
                f"Arm Distribution: {bandit.arm_counts.tolist()}"
            )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds ({episodes / elapsed:.1f} eps/sec)")

    if weights_path:
        w_path = Path(weights_path)
        bandit.save_weights(w_path)
        print(f"Saved trained bandit weights to: {w_path}")

    return bandit


def evaluate_agent(
    agent: Any,
    env: CustomerSupportBanditEnv,
    test_episodes: int = 1000,
    is_bandit: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates an agent on test_episodes unseen cases without weight updates.
    """
    total_reward = 0.0
    financial_losses = 0
    false_rejections = 0
    escalations = 0
    correct_resolutions = 0
    arm_counts = np.zeros(6, dtype=np.int64)

    for _ in range(test_episodes):
        context, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            if is_bandit:
                arm, _ = agent.select_arm(context, explore=False)
            else:
                arm = agent.select_arm(context)

            arm_counts[arm] += 1
            context, reward, done, _, info = env.step(arm)
            ep_reward += reward

            case = env.current_case
            if case is not None:
                # Track key business metrics
                if arm == 0:  # APPROVE
                    if case.is_fraud or not case.has_real_receipt:
                        financial_losses += 1
                    elif case.is_within_window and case.order_exists:
                        correct_resolutions += 1
                elif arm == 1:  # REJECT
                    if not case.is_fraud and case.is_within_window and case.order_exists:
                        false_rejections += 1
                    elif case.is_fraud or not case.is_within_window:
                        correct_resolutions += 1
                elif arm == 5:  # ESCALATE
                    escalations += 1

        total_reward += ep_reward

    return {
        "total_reward": round(total_reward, 2),
        "avg_reward": round(total_reward / test_episodes, 2),
        "financial_loss_rate": round(financial_losses / test_episodes, 4),
        "false_rejection_rate": round(false_rejections / test_episodes, 4),
        "escalation_rate": round(escalations / test_episodes, 4),
        "arm_counts": arm_counts.tolist(),
    }


def run_phase_d_benchmark(bandit: LinUCBBandit, test_episodes: int = 1000):
    """
    Phase D: Run unseen cases across Bandit, Baseline 1, and Baseline 2.
    """
    test_seed = 9999
    print(f"\n=======================================================")
    print(f"Phase D: Final Comparative Evaluation ({test_episodes:,} Unseen Cases)")
    print(f"=======================================================")

    env1 = CustomerSupportBanditEnv(seed=test_seed)
    env2 = CustomerSupportBanditEnv(seed=test_seed)
    env3 = CustomerSupportBanditEnv(seed=test_seed)

    escaler = Baseline1Escaler()
    accommodator = Baseline2Accommodator()

    results_bandit = evaluate_agent(bandit, env1, test_episodes, is_bandit=True)
    results_base1 = evaluate_agent(escaler, env2, test_episodes, is_bandit=False)
    results_base2 = evaluate_agent(accommodator, env3, test_episodes, is_bandit=False)

    print(f"\n{'Metric':<25} | {'LinUCB Bandit':<16} | {'Baseline 1 (Escaler)':<20} | {'Baseline 2 (Accommodator)':<25}")
    print("-" * 95)
    print(f"{'Total Cumulative Reward':<25} | {results_bandit['total_reward']:<+16.2f} | {results_base1['total_reward']:<+20.2f} | {results_base2['total_reward']:<+25.2f}")
    print(f"{'Average Reward/Episode':<25} | {results_bandit['avg_reward']:<+16.2f} | {results_base1['avg_reward']:<+20.2f} | {results_base2['avg_reward']:<+25.2f}")
    print(f"{'Financial Loss Rate':<25} | {results_bandit['financial_loss_rate']:<16.2%} | {results_base1['financial_loss_rate']:<20.2%} | {results_base2['financial_loss_rate']:<25.2%}")
    print(f"{'False Rejection Rate':<25} | {results_bandit['false_rejection_rate']:<16.2%} | {results_base1['false_rejection_rate']:<20.2%} | {results_base2['false_rejection_rate']:<25.2%}")
    print(f"{'Escalation Rate':<25} | {results_bandit['escalation_rate']:<16.2%} | {results_base1['escalation_rate']:<20.2%} | {results_base2['escalation_rate']:<25.2%}")
    print("-" * 95)

    return results_bandit, results_base1, results_base2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Module 4 LinUCB Contextual Bandit")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--test-episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--alpha", type=float, default=0.50, help="LinUCB exploration parameter")
    parser.add_argument("--weights-file", type=str, default=str(_WORKSPACE_ROOT / "module4" / "bandit_weights.npz"), help="Path to save/load bandit weights")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation benchmark directly using existing weights")

    args = parser.parse_args()

    weights_file = Path(args.weights_file)

    if args.eval_only:
        if not weights_file.exists():
            print(f"Error: Weights file not found: {weights_file}")
            sys.exit(1)
        bandit = LinUCBBandit(n_features=CONTEXT_DIMENSION, n_arms=6)
        bandit.load_weights(weights_file)
        print(f"Loaded existing bandit weights from: {weights_file}")
    else:
        bandit = train_linucb_bandit(episodes=args.episodes, alpha=args.alpha, weights_path=weights_file)

    run_phase_d_benchmark(bandit, test_episodes=args.test_episodes)
