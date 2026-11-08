# Module 4 — 54,000-State Tabular Q-Table Trainer
# ===============================================
# Trains the 54,000 x 6 discrete Q-table using offline experience replay
# from our pipeline dataset (or uploaded reference dataset) using Q-Learning.
#
# State Formula:
#   state_id = (((policy_id * 27 + constraint_state) * 4 + evidence_rel) * 10 + turn) * 10 + fraud_prob_bucket
#
# Actions:
#   0: APPROVE (Terminal)
#   1: REJECT (Terminal)
#   2: ASK_FOR_EVIDENCE (Non-terminal)
#   3: PERFORM_VERIFICATION (Non-terminal)
#   4: PROPOSE_ALTERNATIVE (Terminal)
#   5: ESCALATE (Terminal)

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from module4.generate_q_dataset import (
    TOTAL_STATES, NUM_ACTIONS, encode_state, decode_state,
    decode_constraints, ACTION_NAMES
)

logger = logging.getLogger(__name__)


def load_dataset_arrays(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset directly into contiguous NumPy arrays for maximum performance:
    returns (states, actions, rewards, next_states, dones).
    """
    print(f"Loading transitions from {csv_path}...")
    t0 = time.time()

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Find column indices
        s_idx = header.index("state_id")
        a_idx = header.index("action")
        r_idx = header.index("reward")
        ns_idx = header.index("next_state_id")
        d_idx = header.index("done")

        for row in reader:
            states.append(int(row[s_idx]))
            actions.append(int(row[a_idx]))
            rewards.append(float(row[r_idx]))
            next_states.append(int(row[ns_idx]))
            dones.append(int(row[d_idx]))

    states = np.array(states, dtype=np.int32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.int32)
    dones = np.array(dones, dtype=np.int32)

    t1 = time.time()
    print(f"Loaded {len(states):,} transitions in {t1 - t0:.2f}s.")
    return states, actions, rewards, next_states, dones


def train_q_table(
    csv_path: str,
    output_npy: str = "module4/q_table_54k.npy",
    epochs: int = 10,
    lr: float = 0.20,
    gamma: float = 0.90,
) -> np.ndarray:
    """
    Train 54,000 x 6 Tabular Q-Table via Bellman dynamic programming updates.
    """
    states, actions, rewards, next_states, dones = load_dataset_arrays(csv_path)
    N = len(states)

    q_table = np.zeros((TOTAL_STATES, NUM_ACTIONS), dtype=np.float32)

    print(f"Starting Q-Learning training for {epochs} epochs (LR={lr}, Gamma={gamma})...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        max_delta = 0.0

        # Run vectorized or sequential Bellman updates
        # To maintain exact ordering and avoid race conditions, process in mini-chunks or scan
        for i in range(N):
            s = states[i]
            a = actions[i]
            r = rewards[i]
            ns = next_states[i]
            d = dones[i]

            if d or ns < 0 or ns >= TOTAL_STATES:
                target = r
            else:
                target = r + gamma * np.max(q_table[ns])

            td_error = target - q_table[s, a]
            delta = abs(lr * td_error)
            if delta > max_delta:
                max_delta = delta

            q_table[s, a] += lr * td_error

        t_epoch_end = time.time()
        print(f"Epoch {epoch:2d}/{epochs:2d} | Max Q-Delta: {max_delta:8.4f} | Time: {t_epoch_end - t_epoch_start:.2f}s")
        if max_delta < 0.001 and epoch >= 3:
            print(f"Converged early at epoch {epoch}!")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s. Saving Q-table to {output_npy}...")

    os.makedirs(os.path.dirname(os.path.abspath(output_npy)), exist_ok=True)
    np.save(output_npy, q_table)

    # Validate trained policy against key archetypes
    greedy_policy = np.argmax(q_table, axis=1)
    unique_actions, counts = np.unique(greedy_policy, return_counts=True)
    distribution = {ACTION_NAMES[a]: int(c) for a, c in zip(unique_actions, counts)}
    print("Greedy Policy Action Distribution across all 54,000 states:")
    for name, cnt in distribution.items():
        print(f"  {name:22s}: {cnt:5d} states ({cnt / TOTAL_STATES * 100:.1f}%)")

    # Save summary metadata
    meta = {
        "total_states": TOTAL_STATES,
        "num_actions": NUM_ACTIONS,
        "epochs": epochs,
        "learning_rate": lr,
        "discount_factor": gamma,
        "source_dataset": str(csv_path),
        "total_transitions": N,
        "training_time_seconds": round(total_time, 2),
        "greedy_action_distribution": distribution,
    }
    meta_path = output_npy.replace(".npy", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return q_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 54k-state Q-table")
    parser.add_argument("--dataset", type=str, default="module4/q_learning_pipeline_dataset.csv",
                        help="Path to dataset CSV (e.g. q_learning_pipeline_dataset.csv or q_learning_1_2M_instances.csv)")
    parser.add_argument("--out", type=str, default="module4/q_table_54k.npy",
                        help="Output path for trained Q-table .npy")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.20, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.90, help="Discount factor")
    args = parser.parse_args()

    train_q_table(
        csv_path=args.dataset,
        output_npy=args.out,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma
    )
