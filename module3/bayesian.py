# Module 3 — Bayesian Network for Reasoning Under Uncertainty
# ============================================================
# Implements a probabilistic graphical model to estimate:
#   1. P(Fraud | Evidence)  — posterior probability of fraudulent claim
#   2. Evidence reliability — quality and trustworthiness of submitted evidence
#
# Exact Bayesian inference using NumPy via joint enumeration over all hidden states,
# normalized by the marginal partition function.
#
# Network structure:
#
#   [Claim History] ──────────┬─────────────────┐
#         │                   │                 │
#         ▼                   ▼                 ▼
#   [Evidence Quality]   [Temporal Consistency] [Statement Match] ◀── [Threat Behavior]
#         │                   │                 │
#         └───────────────────┼─────────────────┘
#                             ▼
#                          [Fraud]
#
# Nodes:
#   - claim_history in {clean, moderate, suspicious} (Prior: [0.70, 0.20, 0.10])
#   - threat_behavior in {none, mild, aggressive} (Prior: [0.60, 0.25, 0.15])
#   - evidence_quality in {strong, moderate, weak, fabricated} (Parents: claim_history)
#   - temporal_consistency in {aligned, contradictory} (Parents: claim_history)
#   - statement_match in {consistent, minor_issues, contradictory} (Parents: claim_history, threat_behavior)
#   - fraud in {legitimate, suspicious, fraudulent} (Parents: evidence_quality, temporal_consistency, statement_match)

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

try:
    from shared.models import (
        ClaimExtraction, CSPResult, RiskAssessment, ConstraintStatus,
        ThreatLevel, EvidenceStrength, Sentiment
    )
except ImportError:
    from ..shared.models import (
        ClaimExtraction, CSPResult, RiskAssessment, ConstraintStatus,
        ThreatLevel, EvidenceStrength, Sentiment
    )


def _resolve_claim(claim_input: Any) -> ClaimExtraction:
    """Resolve claim input from a ClaimExtraction instance, dict, JSON file path, or text."""
    if isinstance(claim_input, ClaimExtraction):
        return claim_input
    if isinstance(claim_input, dict):
        return ClaimExtraction.from_dict(claim_input)
    if isinstance(claim_input, (str, Path)):
        claim_path = Path(claim_input)
        if claim_path.exists():
            return ClaimExtraction.from_file(claim_path)
        try:
            data = json.loads(str(claim_input))
            if isinstance(data, dict):
                return ClaimExtraction.from_dict(data)
        except Exception:
            pass
        from module4.nlu import NLUExtractor
        return NLUExtractor().extract(str(claim_input))
    raise TypeError(f"Unsupported claim input type: {type(claim_input)}")


def _resolve_csp(csp_input: Any, claim: ClaimExtraction,
                 completed_verifications: Optional[List[str]] = None) -> CSPResult:
    """Resolve CSPResult from instance, dict, JSON file path, or dynamic CSPSolver."""
    if isinstance(csp_input, CSPResult):
        return csp_input
    if isinstance(csp_input, dict):
        return CSPResult.from_dict(csp_input)
    if isinstance(csp_input, (str, Path)):
        csp_path = Path(csp_input)
        if csp_path.exists():
            return CSPResult.from_file(csp_path)
        try:
            data = json.loads(str(csp_input))
            if isinstance(data, dict):
                return CSPResult.from_dict(data)
        except Exception:
            pass
    # If not provided, dynamically solve CSP using policies
    from module1.policy_graph import PolicyGraph
    from module1.csp import CSPSolver
    solver = CSPSolver(PolicyGraph())
    return solver.evaluate(claim, completed_verifications)


class BayesianNode:
    """A single node in the Bayesian Network with states and conditional probability tables."""

    def __init__(self, name: str, states: List[str]):
        self.name = name
        self.states = states
        self.num_states = len(states)
        self.cpt: Optional[np.ndarray] = None
        self.parents: List['BayesianNode'] = []
        self.observed: Optional[int] = None

    def set_prior(self, probs: List[float]):
        """Set prior probability distribution (root node without parents)."""
        self.cpt = np.array(probs, dtype=np.float64)

    def set_cpt(self, cpt: np.ndarray):
        """Set conditional probability table."""
        self.cpt = np.array(cpt, dtype=np.float64)

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

    Consumes UNCERTAIN, VIOLATED, and POTENTIALLY_MANIPULATED flags from Module 1's CSP solver,
    along with NLU-extracted claim semantics, and computes exact posteriors using Bayes' theorem.
    """

    def __init__(self):
        self._build_network()

    def _build_network(self):
        """Construct the 6-node Bayesian Network and conditional probability tables."""

        # 1. Define nodes and state spaces
        self.claim_history = BayesianNode(
            "claim_history", ["clean", "moderate", "suspicious"]
        )
        self.threat_behavior = BayesianNode(
            "threat_behavior", ["none", "mild", "aggressive"]
        )
        self.evidence_quality = BayesianNode(
            "evidence_quality", ["strong", "moderate", "weak", "fabricated"]
        )
        self.temporal_consistency = BayesianNode(
            "temporal_consistency", ["aligned", "contradictory"]
        )
        self.statement_match = BayesianNode(
            "statement_match", ["consistent", "minor_issues", "contradictory"]
        )
        self.fraud = BayesianNode(
            "fraud", ["legitimate", "suspicious", "fraudulent"]
        )

        # 2. Priors for root nodes
        self.claim_history.set_prior([0.70, 0.20, 0.10])
        self.threat_behavior.set_prior([0.60, 0.25, 0.15])

        # 3. P(Evidence Quality | Claim History)
        # Rows: claim_history (clean, moderate, suspicious)
        # Cols: evidence_quality (strong, moderate, weak, fabricated)
        self.evidence_quality.parents = [self.claim_history]
        self.evidence_quality.set_cpt(np.array([
            [0.60, 0.25, 0.10, 0.05],  # clean
            [0.30, 0.35, 0.25, 0.10],  # moderate
            [0.10, 0.20, 0.35, 0.35],  # suspicious
        ]))

        # 4. P(Temporal Consistency | Claim History)
        # Rows: claim_history (clean, moderate, suspicious)
        # Cols: temporal_consistency (aligned, contradictory)
        self.temporal_consistency.parents = [self.claim_history]
        self.temporal_consistency.set_cpt(np.array([
            [0.90, 0.10],  # clean
            [0.70, 0.30],  # moderate
            [0.40, 0.60],  # suspicious
        ]))

        # 5. P(Statement Match | Claim History, Threat Behavior)
        # Shape: (claim_history, threat_behavior, statement_match)
        self.statement_match.parents = [self.claim_history, self.threat_behavior]
        self.statement_match.set_cpt(np.array([
            [  # clean history
                [0.85, 0.12, 0.03],  # none
                [0.65, 0.25, 0.10],  # mild
                [0.45, 0.35, 0.20],  # aggressive
            ],
            [  # moderate history
                [0.55, 0.35, 0.10],  # none
                [0.35, 0.45, 0.20],  # mild
                [0.20, 0.40, 0.40],  # aggressive
            ],
            [  # suspicious history
                [0.30, 0.40, 0.30],  # none
                [0.15, 0.35, 0.50],  # mild
                [0.05, 0.25, 0.70],  # aggressive
            ],
        ]))

        # 6. P(Fraud | Evidence Quality, Temporal Consistency, Statement Match)
        # Shape: (evidence_quality=4, temporal_consistency=2, statement_match=3, fraud=3)
        p_f = np.zeros((4, 2, 3, 3), dtype=np.float64)

        # E=0 (strong)
        # TC=0 (aligned)
        p_f[0, 0, 0] = [0.92, 0.06, 0.02]
        p_f[0, 0, 1] = [0.72, 0.20, 0.08]
        p_f[0, 0, 2] = [0.40, 0.35, 0.25]
        # TC=1 (contradictory)
        p_f[0, 1, 0] = [0.42, 0.38, 0.20]
        p_f[0, 1, 1] = [0.30, 0.40, 0.30]
        p_f[0, 1, 2] = [0.18, 0.37, 0.45]

        # E=1 (moderate)
        # TC=0 (aligned)
        p_f[1, 0, 0] = [0.78, 0.16, 0.06]
        p_f[1, 0, 1] = [0.55, 0.30, 0.15]
        p_f[1, 0, 2] = [0.30, 0.40, 0.30]
        # TC=1 (contradictory)
        p_f[1, 1, 0] = [0.35, 0.40, 0.25]
        p_f[1, 1, 1] = [0.22, 0.43, 0.35]
        p_f[1, 1, 2] = [0.10, 0.35, 0.55]

        # E=2 (weak)
        # TC=0 (aligned)
        p_f[2, 0, 0] = [0.65, 0.25, 0.10]
        p_f[2, 0, 1] = [0.38, 0.37, 0.25]
        p_f[2, 0, 2] = [0.15, 0.40, 0.45]
        # TC=1 (contradictory)
        p_f[2, 1, 0] = [0.25, 0.45, 0.30]
        p_f[2, 1, 1] = [0.15, 0.40, 0.45]
        p_f[2, 1, 2] = [0.05, 0.30, 0.65]

        # E=3 (fabricated)
        # TC=0 (aligned)
        p_f[3, 0, 0] = [0.08, 0.22, 0.70]
        p_f[3, 0, 1] = [0.04, 0.16, 0.80]
        p_f[3, 0, 2] = [0.01, 0.09, 0.90]
        # TC=1 (contradictory)
        p_f[3, 1, 0] = [0.04, 0.16, 0.80]
        p_f[3, 1, 1] = [0.02, 0.10, 0.88]
        p_f[3, 1, 2] = [0.01, 0.04, 0.95]

        self.fraud.parents = [
            self.evidence_quality, self.temporal_consistency, self.statement_match
        ]
        self.fraud.set_cpt(p_f)

    # Backward compatibility alias for statement_consistency
    @property
    def statement_consistency(self) -> BayesianNode:
        return self.statement_match

    @statement_consistency.setter
    def statement_consistency(self, node: BayesianNode):
        self.statement_match = node

    def assess_risk(self, claim: Union[ClaimExtraction, Dict[str, Any], str, Path],
                    csp_result: Optional[Union[CSPResult, Dict[str, Any], str, Path]] = None,
                    completed_verifications: Optional[List[str]] = None) -> RiskAssessment:
        """
        Perform exact Bayesian inference to compute fraud probability and evidence reliability.

        Args:
            claim: Structured claim representation from Module 4 NLU, dict, file path, or text.
            csp_result: Constraint satisfaction result from Module 1 CSP, dict, file path, or None.
            completed_verifications: List of administrative verification checks performed.

        Returns:
            Populated RiskAssessment with marginal posteriors and recommendations.
        """
        resolved_claim = _resolve_claim(claim)
        resolved_csp = _resolve_csp(csp_result, resolved_claim, completed_verifications)

        # Step 1: Set observations from inputs
        self._set_observations(resolved_claim, resolved_csp, completed_verifications or [])

        # Step 2: Compute posteriors via exact joint enumeration
        posteriors = self._compute_posteriors()

        # Step 3: Extract fraud probability = P(suspicious) + P(fraudulent)
        fraud_probs = posteriors["fraud"]
        fraud_probability = float(fraud_probs[1] + fraud_probs[2])

        # Step 4: Compute evidence reliability
        evidence_probs = posteriors["evidence_quality"]
        if not resolved_claim.evidence_provided:
            evidence_reliability = 0.0
        else:
            evidence_reliability = float(
                evidence_probs[0] * 1.0 +
                evidence_probs[1] * 0.7 +
                evidence_probs[2] * 0.3 +
                evidence_probs[3] * 0.0
            )

        # Step 5: Classify evidence strength
        evidence_strength = self._classify_evidence(evidence_probs, resolved_claim, resolved_csp)

        # Step 6: Identify contributing risk factors
        risk_factors = self._identify_risk_factors(resolved_claim, resolved_csp, posteriors)

        # Step 7: Generate recommendation
        recommendation = self._generate_recommendation(
            fraud_probability, evidence_reliability, risk_factors
        )

        temporal_probs = posteriors["temporal_consistency"]
        statement_probs = posteriors["statement_match"]

        posterior_details = {
            "P(legitimate)": round(float(fraud_probs[0]), 4),
            "P(suspicious)": round(float(fraud_probs[1]), 4),
            "P(fraudulent)": round(float(fraud_probs[2]), 4),
            "P(evidence_strong)": round(float(evidence_probs[0]), 4),
            "P(evidence_moderate)": round(float(evidence_probs[1]), 4),
            "P(evidence_weak)": round(float(evidence_probs[2]), 4),
            "P(evidence_fabricated)": round(float(evidence_probs[3]), 4),
            "P(temporal_aligned)": round(float(temporal_probs[0]), 4),
            "P(statement_consistent)": round(float(statement_probs[0]), 4),
        }

        # Step 8: Clear observations for subsequent invocations
        self._clear_observations()

        return RiskAssessment(
            fraud_probability=round(fraud_probability, 4),
            evidence_reliability=round(evidence_reliability, 4),
            evidence_strength=evidence_strength,
            risk_factors=risk_factors,
            recommendation=recommendation,
            posterior_details=posterior_details,
        )

    def assess_risk_file(
        self,
        claim_file: Union[str, Path],
        csp_file: Optional[Union[str, Path]] = None,
        output_file: Optional[Union[str, Path]] = None,
        completed_verifications: Optional[List[str]] = None,
    ) -> RiskAssessment:
        """
        Evaluate risk directly from JSON input files and optionally write output to a JSON file.
        """
        claim_path = Path(claim_file)
        if not claim_path.exists():
            raise FileNotFoundError(f"Claim input file not found: {claim_file}")
        claim = ClaimExtraction.from_file(claim_path)

        csp_result: Optional[CSPResult] = None
        if csp_file is not None and Path(csp_file).exists():
            csp_result = CSPResult.from_file(Path(csp_file))
        else:
            from module1.policy_graph import PolicyGraph
            from module1.csp import CSPSolver
            solver = CSPSolver(PolicyGraph())
            csp_result = solver.evaluate(claim, completed_verifications)
            if csp_file is not None:
                csp_result.to_file(csp_file)

        risk = self.assess_risk(claim, csp_result, completed_verifications)

        if output_file is not None:
            risk.to_file(output_file)

        return risk

    def assess_risk_dict(
        self,
        claim_dict: Dict[str, Any],
        csp_dict: Optional[Dict[str, Any]] = None,
        completed_verifications: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate risk from dictionaries and return the RiskAssessment as a JSON-compatible dict.
        """
        claim = ClaimExtraction.from_dict(claim_dict)
        csp_result = CSPResult.from_dict(csp_dict) if csp_dict else None
        risk = self.assess_risk(claim, csp_result, completed_verifications)
        return risk.to_dict()

    def _set_observations(self, claim: ClaimExtraction, csp_result: CSPResult,
                          completed_verifications: List[str]):
        """Map claim, CSP data, and completed verifications to Bayesian network observations."""

        # 1. Threat behavior observation
        threat_map = {
            ThreatLevel.NONE: "none",
            ThreatLevel.LOW: "mild",
            ThreatLevel.MEDIUM: "mild",
            ThreatLevel.HIGH: "aggressive",
        }
        self.threat_behavior.observe(threat_map.get(claim.threat_level, "none"))

        # 2. Claim history observation
        num_manipulated = sum(
            1 for s in csp_result.constraints.values()
            if s == ConstraintStatus.POTENTIALLY_MANIPULATED
        )
        if csp_result.has_manipulated or num_manipulated > 0:
            self.claim_history.observe("suspicious")
        elif any(
            any(term in key.lower() for term in ("history", "previous_claim", "claim_limit"))
            and status == ConstraintStatus.VIOLATED
            for key, status in csp_result.constraints.items()
        ):
            self.claim_history.observe("suspicious")
        elif claim.claimed_previous_claims is not None and claim.claimed_previous_claims >= 3:
            self.claim_history.observe("suspicious")
        elif sum(1 for s in csp_result.constraints.values() if s == ConstraintStatus.UNCERTAIN) > 2:
            self.claim_history.observe("moderate")
        else:
            self.claim_history.observe("clean")

        # 3. Evidence quality observation
        if csp_result.has_manipulated or "fabricated_evidence" in claim.adversarial_tactics:
            self.evidence_quality.observe("fabricated")
        elif "evidence_check" in completed_verifications:
            self.evidence_quality.observe("strong")
        elif not claim.evidence_provided:
            self.evidence_quality.observe("weak")
        else:
            self.evidence_quality.observe("moderate")

        # 4. Temporal consistency observation
        temporal_violated = False
        temporal_terms = ("window", "date", "day", "time", "temporal", "period", "expiration")
        for key, status in csp_result.constraints.items():
            if status == ConstraintStatus.VIOLATED and any(t in key.lower() for t in temporal_terms):
                temporal_violated = True
                break

        if not temporal_violated:
            for pol in csp_result.violated_policies:
                if any(t in pol.lower() for t in temporal_terms):
                    temporal_violated = True
                    break

        if not temporal_violated and getattr(claim, "contradictory_claims", None):
            if any(c.get("type") == "date_mismatch" for c in claim.contradictory_claims):
                temporal_violated = True

        if temporal_violated:
            self.temporal_consistency.observe("contradictory")
        else:
            self.temporal_consistency.observe("aligned")

        # 5. Statement match observation
        if "contradiction" in claim.adversarial_tactics or bool(getattr(claim, "contradictory_claims", [])):
            self.statement_match.observe("contradictory")
        elif (
            csp_result.has_manipulated or
            "fabricated_evidence" in claim.adversarial_tactics or
            "emotional_appeal" in claim.adversarial_tactics or
            bool(getattr(claim, "emotional_appeals", [])) or
            "policy_gaslighting" in getattr(claim, "persuasive_strategies", [])
        ):
            self.statement_match.observe("minor_issues")
        else:
            self.statement_match.observe("consistent")

    def _compute_posteriors(self) -> Dict[str, np.ndarray]:
        """
        Compute posterior marginal distributions via exact joint enumeration.

        P(H, T, E, TC, S, F) = P(H) * P(T) * P(E|H) * P(TC|H) * P(S|H,T) * P(F|E,TC,S)
        conditioned on all observed variables, normalized by the marginal partition function.
        """
        # Shape: (3, 3, 4, 2, 3, 3)
        joint = (
            self.claim_history.cpt[:, None, None, None, None, None] *
            self.threat_behavior.cpt[None, :, None, None, None, None] *
            self.evidence_quality.cpt[:, None, :, None, None, None] *
            self.temporal_consistency.cpt[:, None, None, :, None, None] *
            self.statement_match.cpt[:, :, None, None, :, None] *
            self.fraud.cpt[None, None, :, :, :, :]
        )

        # Apply observed evidence conditioning
        nodes_with_axes = [
            (self.claim_history, 0),
            (self.threat_behavior, 1),
            (self.evidence_quality, 2),
            (self.temporal_consistency, 3),
            (self.statement_match, 4),
            (self.fraud, 5),
        ]

        for node, axis in nodes_with_axes:
            if node.observed is not None:
                idx = node.observed
                mask = np.zeros(joint.shape[axis], dtype=np.float64)
                mask[idx] = 1.0
                shape = [1] * 6
                shape[axis] = joint.shape[axis]
                joint = joint * mask.reshape(shape)

        partition_function = float(np.sum(joint))
        if partition_function > 0:
            joint = joint / partition_function
        else:
            joint = np.ones_like(joint) / joint.size

        # Marginal posteriors for each node
        p_h = np.sum(joint, axis=(1, 2, 3, 4, 5))
        p_t = np.sum(joint, axis=(0, 2, 3, 4, 5))
        p_e = np.sum(joint, axis=(0, 1, 3, 4, 5))
        p_tc = np.sum(joint, axis=(0, 1, 2, 4, 5))
        p_s = np.sum(joint, axis=(0, 1, 2, 3, 5))
        p_f = np.sum(joint, axis=(0, 1, 2, 3, 4))

        return {
            "claim_history": p_h,
            "threat_behavior": p_t,
            "evidence_quality": p_e,
            "temporal_consistency": p_tc,
            "statement_match": p_s,
            "statement_consistency": p_s,
            "fraud": p_f,
        }

    def _classify_evidence(self, evidence_probs: np.ndarray,
                           claim: ClaimExtraction,
                           csp_result: CSPResult) -> EvidenceStrength:
        """Classify overall evidence strength based on observations and posteriors."""
        # 1. Check for statement and evidence contradictions
        if (
            "contradiction" in claim.adversarial_tactics or
            bool(getattr(claim, "contradictory_claims", []))
        ):
            return EvidenceStrength.CONTRADICTORY

        # 2. Check for deliberate fabrication signals
        if (
            csp_result.has_manipulated or
            "fabricated_evidence" in claim.adversarial_tactics or
            evidence_probs[3] > 0.5
        ):
            return EvidenceStrength.FABRICATED

        # 3. Check for unprovided evidence baseline
        if not claim.evidence_provided:
            return EvidenceStrength.MISSING

        # 4. Map from posterior probability mode
        max_idx = int(np.argmax(evidence_probs))
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
        """Identify specific human-readable risk factors contributing to the assessment."""
        factors = []

        # Customer threat level
        if claim.threat_level in (ThreatLevel.HIGH, ThreatLevel.MEDIUM):
            factors.append(f"Customer threat level: {claim.threat_level.value}")

        # Adversarial tactics detected
        if claim.adversarial_tactics:
            factors.append(f"Adversarial tactics detected: {', '.join(claim.adversarial_tactics)}")
        if getattr(claim, "threats", None):
            factors.append(f"Direct threats identified: {', '.join(claim.threats)}")
        if getattr(claim, "persuasive_strategies", None):
            factors.append(f"Persuasive strategies: {', '.join(claim.persuasive_strategies)}")

        # Policy violations
        if csp_result.violated_policies:
            factors.append(f"Policy violations: {', '.join(csp_result.violated_policies)}")

        # Temporal discrepancies
        temporal_probs = posteriors.get("temporal_consistency", np.array([]))
        if len(temporal_probs) > 1 and temporal_probs[1] > 0.5:
            factors.append("Temporal discrepancy: policy return/cancellation window violated or dates inconsistent")

        # Statement contradictions
        statement_probs = posteriors.get("statement_match", np.array([]))
        if len(statement_probs) > 2 and statement_probs[2] > 0.5:
            factors.append("Statement contradiction: factual claim conflict or cross-turn mismatch detected")

        # Potentially manipulated constraints
        manipulated = [
            k for k, v in csp_result.constraints.items()
            if v == ConstraintStatus.POTENTIALLY_MANIPULATED
        ]
        if manipulated:
            factors.append(f"Potentially manipulated: {', '.join(manipulated)}")

        # High claim amount
        if claim.amount > 500:
            factors.append(f"High claim value: ${claim.amount:.2f}")

        # Missing or unverified evidence
        if not claim.evidence_provided:
            factors.append("No supporting evidence provided")
        else:
            uncertain = [
                k for k, v in csp_result.constraints.items()
                if v == ConstraintStatus.UNCERTAIN and any(t in k.lower() for t in ("evidence", "receipt", "photo"))
            ]
            if uncertain:
                factors.append(f"Unverified evidence constraints: {', '.join(uncertain)}")

        return factors

    def _generate_recommendation(self, fraud_prob: float,
                                 evidence_reliability: float,
                                 risk_factors: List[str]) -> str:
        """Generate operational guidance calibrated at 0.15, 0.30, 0.50, and 0.70 thresholds."""
        if fraud_prob > 0.70:
            return ("HIGH RISK: Strong indicators of fraudulent claim. "
                    "Recommend rejection or escalation to fraud team.")
        elif fraud_prob > 0.50:
            return ("ELEVATED RISK: Multiple suspicious indicators. "
                    "Recommend additional investigation before proceeding.")
        elif fraud_prob > 0.30:
            return ("MODERATE RISK: Some uncertainty present. "
                    "Recommend requesting additional evidence to clarify.")
        elif fraud_prob > 0.15:
            return ("LOW RISK: Minor concerns noted. "
                    "Proceed with standard verification steps.")
        else:
            return ("MINIMAL RISK: Claim appears legitimate. "
                    "Recommend proceeding with approval after standard checks.")

    def _clear_observations(self):
        """Reset all observations for subsequent evaluations."""
        self.claim_history.clear_observation()
        self.threat_behavior.clear_observation()
        self.evidence_quality.clear_observation()
        self.temporal_consistency.clear_observation()
        self.statement_match.clear_observation()
        self.fraud.clear_observation()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Module 3: Bayesian Network Risk Assessor (File-Driven Pipeline)"
    )
    parser.add_argument(
        "--claim-file", "-c",
        type=str,
        default=str(_WORKSPACE_ROOT / "sample_claim_input.json"),
        help="Path to input ClaimExtraction JSON file (default: sample_claim_input.json)",
    )
    parser.add_argument(
        "--csp-file", "-p",
        type=str,
        default=str(_WORKSPACE_ROOT / "sample_csp_result.json"),
        help="Path to input CSPResult JSON file (default: sample_csp_result.json)",
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=str(_WORKSPACE_ROOT / "sample_risk_assessment.json"),
        help="Path to save output RiskAssessment JSON file (default: sample_risk_assessment.json)",
    )
    parser.add_argument(
        "--verifications", "-v",
        nargs="*",
        default=[],
        help="Completed administrative verifications (e.g. evidence_check order_lookup)",
    )

    args = parser.parse_args()

    assessor = BayesianRiskAssessor()
    claim_path = Path(args.claim_file)
    csp_path = Path(args.csp_file)
    output_path = Path(args.output_file)

    if not claim_path.exists():
        print(f"Error: Claim input file not found: {claim_path}")
        sys.exit(1)

    res = assessor.assess_risk_file(
        claim_file=claim_path,
        csp_file=csp_path,
        output_file=output_path,
        completed_verifications=args.verifications,
    )

    print("Module 3 BayesianRiskAssessor execution successful.")
    print(f"Loaded Claim Input:    {claim_path}")
    print(f"Loaded/Evaluated CSP:  {csp_path}")
    print(f"Saved Output Risk:     {output_path}")
    print(f"P(Fraud):              {res.fraud_probability:.2%}")
    print(f"Evidence Reliability:  {res.evidence_reliability:.2%}")
    print(f"Evidence Strength:     {res.evidence_strength.value}")
    print(f"Recommendation:        {res.recommendation}")
