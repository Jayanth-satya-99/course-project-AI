# Module 4 — Policy Pipeline Q-Dataset Generator
# ===============================================
# Generates a large-scale, high-fidelity experience replay dataset
# (S, A, R, S', done, ground_truth_fraud) directly grounded in our
# pipeline's actual policies (policies.json), CSP constraints (Module 1),
# and Bayesian risk distributions (Module 3).
#
# State Space: 54,000 discrete states (state_id in [0, 53999])
# Action Space: 6 discrete actions (0..5)

import argparse
import csv
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Encodings
# ---------------------------------------------------------------------------

POLICY_REFUND = 0       # P_REFUND (Refund Initiation and Processing)
POLICY_RETURN = 1       # P_RETURN / P_EXCHANGE (Unwanted Returns / Replacement)
POLICY_WARRANTY = 2     # P_WARRANTY (Defects & Warranty Claims)
POLICY_CANCEL = 3       # P_CANCELLATION / P_PAYMENT / P_PRICE_MATCH
POLICY_DELIVERY = 4     # P_DELIVERY / P_SHIPMENT (Missing package / Courier delay)

NUM_POLICIES = 5
NUM_CONSTRAINT_STATES = 27  # 3^3 (c1, c2, c3 in {0: Violated, 1: Uncertain, 2: Satisfied})
NUM_EVIDENCE_RELS = 4       # 0: Missing, 1: Weak, 2: Contradictory, 3: Strong
NUM_TURNS = 10              # 0..9
NUM_FRAUD_BUCKETS = 10      # 0..9 (0=0-10%, ..., 9=90-100%)

TOTAL_STATES = NUM_POLICIES * NUM_CONSTRAINT_STATES * NUM_EVIDENCE_RELS * NUM_TURNS * NUM_FRAUD_BUCKETS
assert TOTAL_STATES == 54000, f"Expected 54,000 states, got {TOTAL_STATES}"

ACTION_APPROVE = 0               # Terminal
ACTION_REJECT = 1                # Terminal
ACTION_ASK_FOR_EVIDENCE = 2      # Non-terminal
ACTION_PERFORM_VERIFICATION = 3  # Non-terminal
ACTION_PROPOSE_ALTERNATIVE = 4   # Terminal
ACTION_ESCALATE = 5              # Terminal

NUM_ACTIONS = 6

ACTION_NAMES = {
    ACTION_APPROVE: "APPROVE",
    ACTION_REJECT: "REJECT",
    ACTION_ASK_FOR_EVIDENCE: "ASK_FOR_EVIDENCE",
    ACTION_PERFORM_VERIFICATION: "PERFORM_VERIFICATION",
    ACTION_PROPOSE_ALTERNATIVE: "PROPOSE_ALTERNATIVE",
    ACTION_ESCALATE: "ESCALATE",
}


# ---------------------------------------------------------------------------
# State Encoding and Decoding Functions
# ---------------------------------------------------------------------------

def encode_constraints(c1: int, c2: int, c3: int) -> int:
    """
    Base-3 encoding of 3 canonical policy constraints:
      c1: Order verification & identity record
      c2: Policy window / eligibility window
      c3: Supporting evidence / document verification
    Values: 0 = Violated, 1 = Uncertain, 2 = Satisfied.
    Returns: constraint_state in [0, 26].
    """
    c1 = min(2, max(0, int(c1)))
    c2 = min(2, max(0, int(c2)))
    c3 = min(2, max(0, int(c3)))
    return (c1 * 9) + (c2 * 3) + c3


def decode_constraints(constraint_state: int) -> Tuple[int, int, int]:
    """Decode constraint_state in [0, 26] back to (c1, c2, c3)."""
    c1 = constraint_state // 9
    rem = constraint_state % 9
    c2 = rem // 3
    c3 = rem % 3
    return c1, c2, c3


def encode_state(policy_id: int, constraint_state: int, evidence_rel: int,
                 turn: int, fraud_prob_bucket: int) -> int:
    """
    Map 5 discrete pipeline variables to state_id in [0, 53999]:
      state_id = (((policy_id * 27 + constraint_state) * 4 + evidence_rel) * 10 + turn) * 10 + fraud_prob_bucket
    """
    p = min(NUM_POLICIES - 1, max(0, int(policy_id)))
    cs = min(NUM_CONSTRAINT_STATES - 1, max(0, int(constraint_state)))
    ev = min(NUM_EVIDENCE_RELS - 1, max(0, int(evidence_rel)))
    t = min(NUM_TURNS - 1, max(0, int(turn)))
    fb = min(NUM_FRAUD_BUCKETS - 1, max(0, int(fraud_prob_bucket)))

    return (((p * 27 + cs) * 4 + ev) * 10 + t) * 10 + fb


def decode_state(state_id: int) -> Tuple[int, int, int, int, int]:
    """
    Decode state_id in [0, 53999] into (policy_id, constraint_state, evidence_rel, turn, fraud_prob_bucket).
    """
    bucket = state_id % 10
    rem1 = state_id // 10
    turn = rem1 % 10
    rem2 = rem1 // 10
    evidence_rel = rem2 % 4
    rem3 = rem2 // 4
    constraint_state = rem3 % 27
    policy_id = rem3 // 27
    return policy_id, constraint_state, evidence_rel, turn, bucket


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

def calculate_reward(
    policy_id: int,
    constraint_state: int,
    evidence_rel: int,
    turn: int,
    fraud_prob_bucket: int,
    action: int,
    ground_truth_fraud: int,
    timed_out: bool = False,
) -> float:
    """
    Dispense exact scalar reward based on the specified business rules:
      +100: Correct resolution (Approve genuine with satisfied constraints, or Reject fraud)
      -200: Financial loss (Approving a fraudulent claim)
      -100: Unnecessary rejection (Rejecting a genuine customer)
      -50:  Policy violation (Approving a genuine claim with a violated constraint)
      +60 / -80: Proposing an alternative (+60 if genuine, -80 if fraud)
      +35 / -20: Escalating (+35 if fraud bucket uncertain 3-7; -20 if obvious <=2 or >=8)
      -1:   Per-turn investigation cost for Actions 2 and 3 (-25 if Turn 9 times out)
    """
    c1, c2, c3 = decode_constraints(constraint_state)
    has_violation = (c1 == 0 or c2 == 0 or c3 == 0)
    all_satisfied = (c1 == 2 and c2 == 2 and c3 == 2)

    if action == ACTION_APPROVE:
        if ground_truth_fraud == 1:
            return -200.0  # Financial loss
        else:
            if has_violation:
                return -50.0  # Policy violation
            elif all_satisfied:
                return 100.0  # Correct resolution
            else:
                return -10.0  # Premature approval under unresolved uncertainty

    elif action == ACTION_REJECT:
        if ground_truth_fraud == 1:
            return 100.0  # Correctly caught fraud
        else:
            return -100.0  # Unnecessary rejection of a genuine customer

    elif action in (ACTION_ASK_FOR_EVIDENCE, ACTION_PERFORM_VERIFICATION):
        if timed_out:
            return -25.0  # Turn 9 timeout without resolution penalty
        return -1.0  # Per-turn investigation cost

    elif action == ACTION_PROPOSE_ALTERNATIVE:
        if ground_truth_fraud == 0:
            return 60.0  # Customer delighted with fair store credit
        else:
            return -80.0  # Store credit conceded to a fraudster

    elif action == ACTION_ESCALATE:
        if 3 <= fraud_prob_bucket <= 7:
            return 35.0  # Appropriate escalation of genuine ambiguity
        else:
            return -20.0  # Inappropriate escalation of clear-cut case

    return 0.0


# ---------------------------------------------------------------------------
# Pipeline Simulation Environment
# ---------------------------------------------------------------------------

class PipelineClaimSimulator:
    """
    Simulates multi-turn customer claim interactions using our pipeline's actual
    policy archetypes (policies.json), CSP constraint rules, and Bayesian updates.
    """

    def __init__(self, fraud_ratio: float = 0.40, seed: Optional[int] = None):
        self.fraud_ratio = fraud_ratio
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_state: Optional[int] = None
        self.ground_truth_fraud: int = 0
        self.policy_id: int = 0
        self.c1: int = 2
        self.c2: int = 2
        self.c3: int = 1
        self.evidence_rel: int = 0
        self.turn: int = 0
        self.fraud_prob_bucket: int = 2

    def reset_from_state(self, state_id: int, ground_truth_fraud: Optional[int] = None) -> int:
        """Initialize simulator directly to a specific state for exploration sweeps."""
        self.policy_id, cs, self.evidence_rel, self.turn, self.fraud_prob_bucket = decode_state(state_id)
        self.c1, self.c2, self.c3 = decode_constraints(cs)

        if ground_truth_fraud is not None:
            self.ground_truth_fraud = ground_truth_fraud
        else:
            # Infer plausible latent fraud based on current state attributes
            if self.fraud_prob_bucket >= 8 or self.evidence_rel == 2 or (cs == 0 and self.fraud_prob_bucket >= 6):
                self.ground_truth_fraud = 1
            elif self.fraud_prob_bucket <= 1 and cs == 26 and self.evidence_rel == 3:
                self.ground_truth_fraud = 0
            elif 3 <= self.fraud_prob_bucket <= 7:
                self.ground_truth_fraud = 1 if random.random() < 0.50 else 0
            elif self.fraud_prob_bucket <= 2:
                self.ground_truth_fraud = 0 if random.random() < 0.85 else 1
            else:
                self.ground_truth_fraud = 1 if random.random() < 0.85 else 0

        self.current_state = state_id
        return self.current_state

    def reset(self, policy_id: Optional[int] = None) -> int:
        """Initialize a fresh claim episode with hidden ground truth starting at Turn 0."""
        self.ground_truth_fraud = 1 if random.random() < self.fraud_ratio else 0
        self.policy_id = policy_id if policy_id is not None else random.randint(0, NUM_POLICIES - 1)
        self.turn = 0

        if self.ground_truth_fraud == 0:
            # Genuine Customer Profile
            # Order usually exists in DB
            self.c1 = 2 if random.random() < 0.90 else 1
            # Usually within return window, occasionally expired (borderline genuine)
            self.c2 = 2 if random.random() < 0.85 else 0
            # Evidence initially missing (customer just started chat) or weak
            if random.random() < 0.55:
                self.evidence_rel = 0  # Missing
                self.c3 = 1            # Uncertain
            elif random.random() < 0.35:
                self.evidence_rel = 1  # Weak
                self.c3 = 1            # Uncertain
            else:
                self.evidence_rel = 3  # Already uploaded receipt
                self.c3 = 2            # Satisfied

            # Genuine fraud bucket typically 0..3 (or 4 if high amount/missing evidence)
            if self.evidence_rel == 3 and self.c1 == 2 and self.c2 == 2:
                self.fraud_prob_bucket = random.choice([0, 1, 2])
            else:
                self.fraud_prob_bucket = random.choice([1, 2, 3, 4])

        else:
            # Fraudulent Actor Profile
            # May have fake order or stolen credentials
            self.c1 = 2 if random.random() < 0.45 else random.choice([0, 1])
            # Often claims delivery issue or past window
            self.c2 = 2 if random.random() < 0.50 else 0
            # Evidence missing, weak, or fabricated/contradictory
            rand_ev = random.random()
            if rand_ev < 0.35:
                self.evidence_rel = 0  # Missing
                self.c3 = 1
            elif rand_ev < 0.35:
                self.evidence_rel = 1  # Weak
                self.c3 = 1
            else:
                self.evidence_rel = 2  # Fabricated / Contradictory
                self.c3 = 0            # Violated

            # Fraud bucket typically higher (4..9)
            if self.evidence_rel == 2:
                self.fraud_prob_bucket = random.choice([7, 8, 9])
            else:
                self.fraud_prob_bucket = random.choice([4, 5, 6, 7, 8])

        cs = encode_constraints(self.c1, self.c2, self.c3)
        self.current_state = encode_state(
            self.policy_id, cs, self.evidence_rel, self.turn, self.fraud_prob_bucket
        )
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute chosen arm (0..5), generate transition, and return (next_state_id, reward, done, info).
        """
        assert self.current_state is not None, "Call reset() before step()"

        cs = encode_constraints(self.c1, self.c2, self.c3)

        # 1. Terminal Actions
        if action in (ACTION_APPROVE, ACTION_REJECT, ACTION_PROPOSE_ALTERNATIVE, ACTION_ESCALATE):
            reward = calculate_reward(
                self.policy_id, cs, self.evidence_rel, self.turn,
                self.fraud_prob_bucket, action, self.ground_truth_fraud, timed_out=False
            )
            return -1, reward, True, {"ground_truth_fraud": self.ground_truth_fraud}

        # 2. Non-terminal Actions
        next_turn = self.turn + 1
        timed_out = (next_turn >= NUM_TURNS)

        if timed_out:
            reward = calculate_reward(
                self.policy_id, cs, self.evidence_rel, self.turn,
                self.fraud_prob_bucket, action, self.ground_truth_fraud, timed_out=True
            )
            return -1, reward, True, {"ground_truth_fraud": self.ground_truth_fraud, "timed_out": True}

        # Execute Non-terminal Action Dynamics
        if action == ACTION_ASK_FOR_EVIDENCE:
            # Customer asked to supply proof / receipt / photo
            if self.ground_truth_fraud == 0:
                # Genuine customer responds
                if random.random() < 0.85:
                    self.evidence_rel = 3  # Strong
                    self.c3 = 2            # Satisfied
                    # Fraud probability drops
                    self.fraud_prob_bucket = max(0, self.fraud_prob_bucket - random.randint(1, 3))
                else:
                    self.evidence_rel = 1  # Customer has blurry receipt
                    self.c3 = 1
                    self.fraud_prob_bucket = min(9, max(0, self.fraud_prob_bucket + random.choice([-1, 0])))
            else:
                # Fraud customer responds
                r = random.random()
                if r < 0.50:
                    # Uploads edited / fabricated evidence
                    self.evidence_rel = 2  # Contradictory / Fabricated
                    self.c3 = 0            # Violated
                    self.fraud_prob_bucket = min(9, self.fraud_prob_bucket + random.randint(2, 4))
                elif r < 0.85:
                    # Stalls or claims cannot find receipt
                    self.evidence_rel = 0  # Missing
                    self.c3 = 1
                    self.fraud_prob_bucket = min(9, self.fraud_prob_bucket + random.randint(1, 2))
                else:
                    # Submits plausible fake
                    self.evidence_rel = 1
                    self.fraud_prob_bucket = min(9, max(3, self.fraud_prob_bucket))

        elif action == ACTION_PERFORM_VERIFICATION:
            # Internal verification check (Carrier API, warehouse log, order database)
            if self.ground_truth_fraud == 0:
                # Genuine verification succeeds
                self.c1 = 2  # Order verified
                if self.c2 != 0:
                    self.c2 = 2  # Policy window confirmed
                self.fraud_prob_bucket = max(0, self.fraud_prob_bucket - random.randint(1, 2))
            else:
                # Fraud verification exposes anomaly
                if random.random() < 0.70:
                    self.c1 = 0  # Address mismatch / order canceled / carrier delivered successfully
                else:
                    self.c2 = 0  # Claim past eligible deadline
                self.fraud_prob_bucket = min(9, self.fraud_prob_bucket + random.randint(2, 3))

        self.turn = next_turn
        new_cs = encode_constraints(self.c1, self.c2, self.c3)
        self.current_state = encode_state(
            self.policy_id, new_cs, self.evidence_rel, self.turn, self.fraud_prob_bucket
        )
        reward = -1.0  # Standard investigation cost

        return self.current_state, reward, False, {"ground_truth_fraud": self.ground_truth_fraud}


# ---------------------------------------------------------------------------
# Dataset Generation Pipeline
# ---------------------------------------------------------------------------

def generate_pipeline_dataset(
    num_episodes: int = 150000,
    output_path: Optional[str] = None,
    fraud_ratio: float = 0.40,
    seed: int = 42,
    enable_full_state_sweep: bool = True,
) -> List[Tuple]:
    """
    Generate an offline experience replay dataset spanning the 54,000 states.
    Phase 1: Multi-turn customer dialogue episodes.
    Phase 2: Systematic state-space sweeps across all 54,000 states to ensure complete coverage.
    """
    logger.info("Initializing Pipeline Claim Simulator (Episodes: %d, Fraud Ratio: %.2f)...",
                num_episodes, fraud_ratio)

    env = PipelineClaimSimulator(fraud_ratio=fraud_ratio, seed=seed)
    dataset: List[Tuple] = []
    visited_states = set()
    action_counts = {a: 0 for a in range(NUM_ACTIONS)}

    # Phase 1: Natural multi-turn customer dialogues
    for ep in range(num_episodes):
        state_id = env.reset()
        visited_states.add(state_id)
        done = False

        while not done:
            p, cs, ev, turn, fb = decode_state(state_id)
            c1, c2, c3 = decode_constraints(cs)

            # Behavioral policy: 50% guided expert exploration + 50% uniform random
            if random.random() < 0.50:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                if c1 == 0 or c2 == 0 or ev == 2 or fb >= 8:
                    action = ACTION_REJECT if random.random() < 0.70 else random.choice([ACTION_ESCALATE, ACTION_PROPOSE_ALTERNATIVE])
                elif c1 == 2 and c2 == 2 and c3 == 2 and ev == 3 and fb <= 2:
                    action = ACTION_APPROVE if random.random() < 0.80 else random.choice([ACTION_PERFORM_VERIFICATION, ACTION_ASK_FOR_EVIDENCE])
                elif ev in (0, 1) and turn < 3:
                    action = ACTION_ASK_FOR_EVIDENCE if random.random() < 0.65 else ACTION_PERFORM_VERIFICATION
                elif fb in (4, 5, 6) and turn >= 2:
                    action = ACTION_ESCALATE if random.random() < 0.60 else ACTION_PROPOSE_ALTERNATIVE
                else:
                    action = random.choice([ACTION_ASK_FOR_EVIDENCE, ACTION_PERFORM_VERIFICATION, ACTION_PROPOSE_ALTERNATIVE])

            next_state_id, reward, done, info = env.step(action)
            action_counts[action] += 1

            record = (
                state_id,
                p,
                cs,
                ev,
                turn,
                fb,
                action,
                int(reward),
                next_state_id,
                1 if done else 0,
                info["ground_truth_fraud"]
            )
            dataset.append(record)

            if not done:
                state_id = next_state_id
                visited_states.add(state_id)

    # Phase 2: Systematic state-space sweep across all 54,000 states
    if enable_full_state_sweep:
        print("Executing systematic state-space sweep across all 54,000 states (all 6 actions)...")
        for s in range(TOTAL_STATES):
            visited_states.add(s)
            p, cs, ev, turn, fb = decode_state(s)

            # Prioritize realistic latent fraud assignment:
            if fb >= 8 or ev == 2 or (cs == 0 and fb >= 6):
                fraud_candidates = [1]
            elif fb <= 1 and cs == 26 and ev == 3:
                fraud_candidates = [0]
            elif 3 <= fb <= 7:
                fraud_candidates = [0, 1]
            elif fb <= 2:
                fraud_candidates = [0] if random.random() < 0.85 else [1]
            else:
                fraud_candidates = [1] if random.random() < 0.85 else [0]

            for gtf in fraud_candidates:
                for action in range(NUM_ACTIONS):
                    env.reset_from_state(s, ground_truth_fraud=gtf)
                    next_state_id, reward, done, info = env.step(action)
                    action_counts[action] += 1
                    dataset.append((
                        s, p, cs, ev, turn, fb, action,
                        int(reward), next_state_id, 1 if done else 0, gtf
                    ))

    # Log statistics
    coverage = (len(visited_states) / TOTAL_STATES) * 100
    print(f"Dataset generated: {len(dataset):,} transitions from {num_episodes:,} episodes + sweeps.")
    print(f"Unique states visited: {len(visited_states):,} / {TOTAL_STATES:,} ({coverage:.2f}% coverage).")
    print("Action distribution:", {ACTION_NAMES[a]: action_counts[a] for a in range(NUM_ACTIONS)})

    # Write to CSV if path provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "state_id", "policy_id", "constraint_state", "evidence_rel",
                "turn", "fraud_prob_bucket", "action", "reward",
                "next_state_id", "done", "ground_truth_fraud"
            ])
            writer.writerows(dataset)
        print(f"Saved dataset to {output_path} ({os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q-Learning Experience Replay Dataset")
    parser.add_argument("--episodes", type=int, default=150000, help="Number of episodes to simulate")
    parser.add_argument("--out", type=str, default="module4/q_learning_pipeline_dataset.csv", help="Output CSV path")
    parser.add_argument("--fraud_ratio", type=float, default=0.40, help="Ratio of fraudulent claims")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_sweep", action="store_true", help="Disable state space sweep")
    args = parser.parse_args()

    generate_pipeline_dataset(
        num_episodes=args.episodes,
        output_path=args.out,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
        enable_full_state_sweep=not args.no_sweep
    )
