# Module 3 — Bayesian Network for Reasoning Under Uncertainty
# ============================================================
# Implements a probabilistic graphical model to estimate:
#   1. P(Fraud | Evidence)  — likelihood of a fraudulent claim
#   2. Evidence reliability  — quality and trustworthiness of submitted evidence
#
# Uses Bayes' theorem to update beliefs as new evidence arrives.
# Distinguishes between: missing, weak, contradictory, and fabricated evidence.
#
# Network structure:
#
#   [Claim History] ──┐
#                     ├──▶ [Fraud] ──▶ [Decision Risk]
#   [Evidence Quality]┘       │
#                             ▼
#                     [Statement Consistency]
#
# All probabilities are computed from scratch using numpy — no external
# probabilistic library required.

import numpy as np
from typing import Dict, List, Optional, Tuple

from shared.models import (
    ClaimExtraction, CSPResult, RiskAssessment, ConstraintStatus,
    ThreatLevel, EvidenceStrength
)


class BayesianNode:
    """A single node in the Bayesian Network with a conditional probability table."""

    def __init__(self, name: str, states: List[str]):
        self.name = name
        self.states = states
        self.num_states = len(states)
        self.cpt: Optional[np.ndarray] = None  # Conditional Probability Table
        self.parents: List['BayesianNode'] = []
        self.observed: Optional[int] = None  # index of observed state, or None

    def set_prior(self, probs: List[float]):
        """Set prior probabilities (root node, no parents)."""
        self.cpt = np.array(probs)

    def set_cpt(self, cpt: np.ndarray):
        """
        Set conditional probability table.
        Shape: (parent1_states, parent2_states, ..., self_states)
        """
        self.cpt = cpt

    def observe(self, state: str):
        """Set this node to an observed state."""
        if state in self.states:
            self.observed = self.states.index(state)

    def clear_observation(self):
        """Remove any observation from this node."""
        self.observed = None


class BayesianRiskAssessor:
    """
    Bayesian Network for fraud probability estimation and evidence reliability assessment.

    Consumes UNCERTAIN and POTENTIALLY_MANIPULATED flags from Module 1's CSP solver,
    along with the extracted claim data, and computes posterior probabilities using
    Bayes' theorem.

    Key principle: Unusual ≠ Fraudulent.
    The model avoids treating uncommon requests as automatically suspicious.
    """

    def __init__(self):
        self._build_network()

    def _build_network(self):
        """Construct the Bayesian Network nodes and probability tables."""

        # ---- Node definitions ----
        self.claim_history = BayesianNode(
            "claim_history", ["clean", "moderate", "suspicious"]
        )
        self.evidence_quality = BayesianNode(
            "evidence_quality", ["strong", "moderate", "weak", "fabricated"]
        )
        self.statement_consistency = BayesianNode(
            "statement_consistency", ["consistent",
                                      "minor_issues", "contradictory"]
        )
        self.threat_behavior = BayesianNode(
            "threat_behavior", ["none", "mild", "aggressive"]
        )
        self.fraud = BayesianNode(
            "fraud", ["legitimate", "suspicious", "fraudulent"]
        )

        # ---- Prior probabilities (root nodes) ----
        # Base rate: most claims are legitimate
        self.claim_history.set_prior([0.70, 0.20, 0.10])
        self.threat_behavior.set_prior([0.60, 0.25, 0.15])

        # ---- Conditional probability tables ----

        # P(Evidence Quality | Claim History)
        # Rows: claim_history states, Cols: evidence_quality states
        self.evidence_quality.parents = [self.claim_history]
        self.evidence_quality.set_cpt(np.array([
            # clean history  → likely strong evidence
            [0.60, 0.25, 0.10, 0.05],
            # moderate history → mixed evidence
            [0.30, 0.35, 0.25, 0.10],
            # suspicious history → likely weak/fabricated
            [0.10, 0.20, 0.35, 0.35],
        ]))

        # P(Statement Consistency | Claim History, Threat Behavior)
        # Shape: (claim_history, threat_behavior, consistency)
        self.statement_consistency.parents = [
            self.claim_history, self.threat_behavior]
        self.statement_consistency.set_cpt(np.array([
            [  # clean history
                [0.80, 0.15, 0.05],  # no threats
                [0.60, 0.30, 0.10],  # mild threats
                [0.40, 0.35, 0.25],  # aggressive threats
            ],
            [  # moderate history
                [0.50, 0.35, 0.15],
                [0.35, 0.40, 0.25],
                [0.20, 0.35, 0.45],
            ],
            [  # suspicious history
                [0.25, 0.40, 0.35],
                [0.15, 0.30, 0.55],
                [0.10, 0.20, 0.70],
            ],
        ]))

        # P(Fraud | Evidence Quality, Statement Consistency)
        # Shape: (evidence_quality, statement_consistency, fraud)
        self.fraud.parents = [
            self.evidence_quality, self.statement_consistency]
        self.fraud.set_cpt(np.array([
            [  # strong evidence
                [0.90, 0.08, 0.02],  # consistent statements
                [0.70, 0.22, 0.08],  # minor issues
                [0.45, 0.35, 0.20],  # contradictory
            ],
            [  # moderate evidence
                [0.65, 0.25, 0.10],
                [0.40, 0.35, 0.25],
                [0.20, 0.35, 0.45],
            ],
            [  # weak evidence
                [0.65, 0.25, 0.10],
                [0.15, 0.30, 0.55],
                [0.08, 0.22, 0.70],
            ],
            [  # fabricated evidence
                [0.10, 0.25, 0.65],
                [0.05, 0.15, 0.80],
                [0.02, 0.08, 0.90],
            ],
        ]))

    def assess_risk(self, claim: ClaimExtraction,
                    csp_result: CSPResult,
                    completed_verifications: Optional[List[str]] = None) -> RiskAssessment:
        """
        Perform Bayesian inference to compute fraud probability and evidence reliability.

        This method:
        1. Maps claim and CSP data to Bayesian network observations
        2. Runs belief propagation to compute posterior P(Fraud | Evidence)
        3. Classifies evidence strength
        4. Identifies specific risk factors
        5. Generates a recommendation

        Args:
            claim: Structured claim from Module 4 NLU.
            csp_result: Constraint evaluation from Module 1 CSP.

        Returns:
            RiskAssessment with fraud probability, evidence reliability, and risk factors.
        """
        # Step 1: Set observations based on claim data
        self._set_observations(
            claim, csp_result, completed_verifications or [])

        # Step 2: Compute posterior probabilities via enumeration
        posteriors = self._compute_posteriors()

        # Step 3: Extract fraud probability
        fraud_probs = posteriors["fraud"]
        # P(fraud) = P(suspicious) + P(fraudulent)
        fraud_probability = fraud_probs[1] + fraud_probs[2]

        # Step 4: Compute evidence reliability
        evidence_probs = posteriors["evidence_quality"]
        evidence_reliability = (evidence_probs[0] * 1.0 +
                                evidence_probs[1] * 0.7 +
                                evidence_probs[2] * 0.3 +
                                evidence_probs[3] * 0.0)

        # Step 5: Classify evidence strength
        evidence_strength = self._classify_evidence(
            evidence_probs, claim, csp_result)

        # Step 6: Identify risk factors
        risk_factors = self._identify_risk_factors(
            claim, csp_result, posteriors)

        # Step 7: Generate recommendation
        recommendation = self._generate_recommendation(
            fraud_probability, evidence_reliability, risk_factors
        )

        # Clear observations for next call
        self._clear_observations()

        return RiskAssessment(
            fraud_probability=round(fraud_probability, 4),
            evidence_reliability=round(evidence_reliability, 4),
            evidence_strength=evidence_strength,
            risk_factors=risk_factors,
            recommendation=recommendation,
            posterior_details={
                "P(legitimate)": round(fraud_probs[0], 4),
                "P(suspicious)": round(fraud_probs[1], 4),
                "P(fraudulent)": round(fraud_probs[2], 4),
                "P(evidence_strong)": round(evidence_probs[0], 4),
                "P(evidence_moderate)": round(evidence_probs[1], 4),
                "P(evidence_weak)": round(evidence_probs[2], 4),
                "P(evidence_fabricated)": round(evidence_probs[3], 4),
            }
        )

    def _set_observations(self, claim: ClaimExtraction, csp_result: CSPResult,
                          completed_verifications: List[str]):
        """Map claim and CSP data to Bayesian network observations."""

        # Claim history observation
        num_uncertain = sum(
            1 for s in csp_result.constraints.values()
            if s == ConstraintStatus.UNCERTAIN
        )
        num_manipulated = sum(
            1 for s in csp_result.constraints.values()
            if s == ConstraintStatus.POTENTIALLY_MANIPULATED
        )

        if num_manipulated > 0:
            self.claim_history.observe("suspicious")
        elif any(
            "history" in key and status == ConstraintStatus.VIOLATED
            for key, status in csp_result.constraints.items()
        ):
            self.claim_history.observe("suspicious")
        elif num_uncertain > 2:
            self.claim_history.observe("moderate")
        else:
            self.claim_history.observe("clean")

        # Threat behavior observation
        threat_map = {
            ThreatLevel.NONE: "none",
            ThreatLevel.LOW: "mild",
            ThreatLevel.MEDIUM: "mild",
            ThreatLevel.HIGH: "aggressive",
        }
        self.threat_behavior.observe(
            threat_map.get(claim.threat_level, "none"))

        if csp_result.has_manipulated or "fabricated_evidence" in claim.adversarial_tactics:
            self.evidence_quality.observe("fabricated")
        elif not claim.evidence_provided:
            self.evidence_quality.observe("weak")
        elif "evidence_check" in completed_verifications:
            self.evidence_quality.observe("strong")
        else:
            self.evidence_quality.observe("moderate")

        if "contradiction" in claim.adversarial_tactics:
            self.statement_consistency.observe("contradictory")
        elif csp_result.has_manipulated:
            self.statement_consistency.observe("minor_issues")
        else:
            self.statement_consistency.observe("consistent")

    def _compute_posteriors(self) -> Dict[str, np.ndarray]:
        """
        Compute posterior probabilities using exact enumeration.

        For our compact network, we enumerate all combinations of hidden
        variables and sum out using Bayes' theorem.
        """
        # Get priors / observations
        p_history = self._get_distribution(self.claim_history)
        p_threat = self._get_distribution(self.threat_behavior)

        # Enumerate the joint distribution so evidence and consistency remain
        # conditioned on their shared parent nodes.
        p_evidence = np.zeros(self.evidence_quality.num_states)
        p_consistency = np.zeros(self.statement_consistency.num_states)
        p_fraud = np.zeros(self.fraud.num_states)
        total = 0.0
        for h in range(self.claim_history.num_states):
            for t in range(self.threat_behavior.num_states):
                for e in range(self.evidence_quality.num_states):
                    if self.evidence_quality.observed is not None and \
                       e != self.evidence_quality.observed:
                        continue
                    for c in range(self.statement_consistency.num_states):
                        if self.statement_consistency.observed is not None and \
                           c != self.statement_consistency.observed:
                            continue
                        probability = (
                            p_history[h] * p_threat[t] *
                            self.evidence_quality.cpt[h, e] *
                            self.statement_consistency.cpt[h, t, c]
                        )
                        total += probability
                        p_evidence[e] += probability
                        p_consistency[c] += probability
                        p_fraud += probability * self.fraud.cpt[e, c]

        if total > 0:
            p_fraud /= total
            p_evidence /= total
            p_consistency /= total

        return {
            "fraud": p_fraud,
            "evidence_quality": p_evidence,
            "statement_consistency": p_consistency,
        }

    def _get_distribution(self, node: BayesianNode) -> np.ndarray:
        """Get the probability distribution for a node (observed or prior)."""
        if node.observed is not None:
            dist = np.zeros(node.num_states)
            dist[node.observed] = 1.0
            return dist
        return node.cpt.copy() if node.cpt.ndim == 1 else np.ones(node.num_states) / node.num_states

    def _classify_evidence(self, evidence_probs: np.ndarray,
                           claim: ClaimExtraction,
                           csp_result: CSPResult) -> EvidenceStrength:
        """Classify overall evidence strength based on posterior probabilities."""
        # Check for fabrication signals
        if csp_result.has_manipulated:
            return EvidenceStrength.FABRICATED

        # Check for contradictions in the claim
        if "contradiction" in claim.adversarial_tactics:
            return EvidenceStrength.CONTRADICTORY

        # Check if evidence is missing
        if not claim.evidence_provided:
            return EvidenceStrength.MISSING

        # Use posterior probabilities
        max_idx = np.argmax(evidence_probs)
        strength_map = {
            0: EvidenceStrength.STRONG,
            1: EvidenceStrength.MODERATE,
            2: EvidenceStrength.WEAK,
            3: EvidenceStrength.FABRICATED,
        }
        return strength_map.get(max_idx, EvidenceStrength.MODERATE)

    def _identify_risk_factors(self, claim: ClaimExtraction,
                               csp_result: CSPResult,
                               posteriors: Dict[str, np.ndarray]) -> List[str]:
        """Identify specific risk factors contributing to the fraud assessment."""
        factors = []

        # High threat level
        if claim.threat_level in (ThreatLevel.HIGH, ThreatLevel.MEDIUM):
            factors.append(
                f"Customer threat level: {claim.threat_level.value}")

        # Adversarial tactics detected
        if claim.adversarial_tactics:
            tactics_str = ", ".join(claim.adversarial_tactics)
            factors.append(f"Adversarial tactics detected: {tactics_str}")

        # CSP violations
        if csp_result.violated_policies:
            factors.append(
                f"Policy violations: {', '.join(csp_result.violated_policies)}"
            )

        # Potentially manipulated constraints
        manipulated = [k for k, v in csp_result.constraints.items()
                       if v == ConstraintStatus.POTENTIALLY_MANIPULATED]
        if manipulated:
            factors.append(
                f"Potentially manipulated: {', '.join(manipulated)}"
            )

        # High claim amount
        if claim.amount > 500:
            factors.append(f"High claim value: ${claim.amount:.2f}")

        # Missing evidence
        uncertain = [k for k, v in csp_result.constraints.items()
                     if v == ConstraintStatus.UNCERTAIN]
        if uncertain:
            factors.append(
                f"Unverified constraints: {', '.join(uncertain)}"
            )

        # No evidence at all
        if not claim.evidence_provided:
            factors.append("No supporting evidence provided")

        # Statement consistency concerns
        consistency_probs = posteriors.get(
            "statement_consistency", np.array([]))
        if len(consistency_probs) > 2 and consistency_probs[2] > 0.3:
            factors.append(
                f"Statement consistency concern: "
                f"P(contradictory) = {consistency_probs[2]:.2f}"
            )

        return factors

    def _generate_recommendation(self, fraud_prob: float,
                                 evidence_reliability: float,
                                 risk_factors: List[str]) -> str:
        """Generate a textual recommendation based on the risk assessment."""
        if fraud_prob > 0.7:
            return ("HIGH RISK: Strong indicators of fraudulent claim. "
                    "Recommend rejection or escalation to fraud team.")
        elif fraud_prob > 0.5:
            return ("ELEVATED RISK: Multiple suspicious indicators. "
                    "Recommend additional investigation before proceeding.")
        elif fraud_prob > 0.3:
            return ("MODERATE RISK: Some uncertainty present. "
                    "Recommend requesting additional evidence to clarify.")
        elif fraud_prob > 0.15:
            return ("LOW RISK: Minor concerns noted. "
                    "Proceed with standard verification steps.")
        else:
            return ("MINIMAL RISK: Claim appears legitimate. "
                    "Recommend proceeding with approval after standard checks.")

    def _clear_observations(self):
        """Reset all observations for the next assessment."""
        self.claim_history.clear_observation()
        self.threat_behavior.clear_observation()
        self.evidence_quality.clear_observation()
        self.statement_consistency.clear_observation()
        self.fraud.clear_observation()
