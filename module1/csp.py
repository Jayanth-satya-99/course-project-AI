# Module 1 — Constraint Satisfaction Problem (CSP) Solver
# =======================================================
# Maps extracted customer claims against the policy graph and evaluates
# every required constraint, tagging each as:
#   SATISFIED | VIOLATED | UNCERTAIN | POTENTIALLY_MANIPULATED
#
# When a constraint is UNCERTAIN or POTENTIALLY_MANIPULATED, the CSP solver
# does NOT guess — it triggers Module 3 (Bayesian) for quantitative risk assessment.

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from shared.models import (
    ClaimExtraction, CSPResult, ConstraintStatus, ThreatLevel
)
from .evidence_graph import EvidenceGraph, EvidenceGraphSearchResult
from .policy_graph import PolicyGraph, Policy


class CSPSolver:
    """
    Constraint Satisfaction Problem solver for policy compliance evaluation.

    Variables: Each constraint from the applicable policy clauses.
    Domains:   {SATISFIED, VIOLATED, UNCERTAIN, POTENTIALLY_MANIPULATED}
    Constraints: Defined by policy clauses (temporal, evidence, financial, etc.)
    """

    def __init__(self, policy_graph: PolicyGraph):
        self.policy_graph = policy_graph
        self.evidence_graph = EvidenceGraph(policy_graph)
        # Simulated customer database for history checks
        self._customer_db = {
            "default": {
                "previous_claims_6mo": 1,
                "is_vip": False,
                "account_age_days": 365,
                "order_history": {
                    "ORD-001": {"amount": 500, "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"), "item": "TV",
                                "shipped": True, "delivered": True},
                    "ORD-002": {"amount": 79.99, "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"), "item": "Headphones",
                                "shipped": True, "delivered": True},
                    "ORD-003": {"amount": 1200, "date": datetime.now().strftime("%Y-%m-%d"), "item": "Laptop",
                                "shipped": False, "delivered": False},
                }
            }
        }

    def get_order_record(self, order_id: str) -> Optional[Dict]:
        """Return a copy of the demo transaction record for an order ID."""
        record = self._customer_db["default"]["order_history"].get(order_id)
        return dict(record) if record else None

    def evaluate(self, claim: ClaimExtraction,
                 completed_verifications: Optional[List[str]] = None,
                 evidence_graph_result: Optional[EvidenceGraphSearchResult] = None
                 ) -> CSPResult:
        """
        Run the CSP solver against applicable policies for the given claim.

        Args:
            claim: Structured extraction from Module 4 NLU.
            completed_verifications: List of verification steps already completed.

        Returns:
            CSPResult with every constraint tagged and applicable policies listed.
        """
        if completed_verifications is None:
            completed_verifications = []

        order_record = (
            self.get_order_record(claim.order_id) if claim.order_id else None
        )
        if evidence_graph_result is None:
            evidence_graph_result = self.evidence_graph.search(
                claim, completed_verifications, order_record
            )

        # Search the policy graph, then evaluate only constraints in that subgraph.
        applicable_policies = self.policy_graph.search_applicable_policies(claim.intent)
        constraint_results: Dict[str, ConstraintStatus] = {}
        applicable_policy_ids = [policy.policy_id for policy in applicable_policies]
        violated_policy_ids = []
        uncertain_constraints = []

        for constraint_def in self.policy_graph.search_relevant_constraints(
            applicable_policies
        ):
            clause_id = constraint_def.get("_clause_id", "")
            key = f"{clause_id}:{constraint_def.get('description', clause_id)}"
            status = self._evaluate_json_constraint(
                constraint_def, claim, completed_verifications,
                evidence_graph_result, order_record,
            )
            constraint_results[key] = status

            if status == ConstraintStatus.VIOLATED:
                violated_policy_ids.append(clause_id)
            elif status in (ConstraintStatus.UNCERTAIN,
                            ConstraintStatus.POTENTIALLY_MANIPULATED):
                uncertain_constraints.append(key)

        if not applicable_policies:
            key = "POLICY_NOT_FOUND:request"
            constraint_results[key] = ConstraintStatus.UNCERTAIN
            uncertain_constraints.append(key)

        has_uncertain = any(
            s in (ConstraintStatus.UNCERTAIN, ConstraintStatus.POTENTIALLY_MANIPULATED)
            for s in constraint_results.values()
        )
        has_manipulated = any(
            s == ConstraintStatus.POTENTIALLY_MANIPULATED
            for s in constraint_results.values()
        )

        # Build human-readable explanation
        explanation = self._build_explanation(constraint_results, applicable_policy_ids)

        return CSPResult(
            constraints=constraint_results,
            applicable_policies=applicable_policy_ids,
            violated_policies=list(set(violated_policy_ids)),
            uncertain_constraints=uncertain_constraints,
            has_uncertain=has_uncertain,
            has_manipulated=has_manipulated,
            explanation=explanation,
            evidence_graph=evidence_graph_result.to_dict(),
        )

    def _evaluate_json_constraint(
        self,
        constraint: Dict[str, Any],
        claim: ClaimExtraction,
        completed_verifications: List[str],
        evidence_graph_result: EvidenceGraphSearchResult,
        order_record: Optional[Dict[str, Any]],
    ) -> ConstraintStatus:
        """Evaluate the JSON policy's recursive all/any/conditional predicates."""
        kind = constraint.get("type")
        if kind in {"all", "any"}:
            children = constraint.get("conditions", [])
            if not isinstance(children, list) or not children:
                return ConstraintStatus.UNCERTAIN
            statuses = [
                self._evaluate_json_constraint(
                    child, claim, completed_verifications,
                    evidence_graph_result, order_record,
                )
                for child in children
            ]
            if kind == "all":
                if ConstraintStatus.VIOLATED in statuses:
                    return ConstraintStatus.VIOLATED
                if all(status == ConstraintStatus.SATISFIED for status in statuses):
                    return ConstraintStatus.SATISFIED
                return ConstraintStatus.UNCERTAIN
            if ConstraintStatus.SATISFIED in statuses:
                return ConstraintStatus.SATISFIED
            if all(status == ConstraintStatus.VIOLATED for status in statuses):
                return ConstraintStatus.VIOLATED
            return ConstraintStatus.UNCERTAIN

        if kind == "conditional":
            condition = self._evaluate_comparison(
                constraint.get("if", {}),
                claim, completed_verifications,
                evidence_graph_result, order_record,
            )
            if condition == ConstraintStatus.VIOLATED:
                return ConstraintStatus.SATISFIED
            if condition != ConstraintStatus.SATISFIED:
                return ConstraintStatus.UNCERTAIN
            return self._evaluate_json_constraint(
                constraint.get("then", {}),
                claim, completed_verifications,
                evidence_graph_result, order_record,
            )

        if kind == "presence":
            field_name = constraint.get("field")
            value = self._resolve_policy_field(
                field_name, claim, completed_verifications,
                evidence_graph_result, order_record,
            )
            if value is None:
                return ConstraintStatus.UNCERTAIN
            operator = constraint.get("operator", "is_not_null")
            present = value is not None and value != ""
            if operator in {"is_not_null", "is_not_empty"}:
                return (
                    ConstraintStatus.SATISFIED if present
                    else ConstraintStatus.VIOLATED
                )
            return ConstraintStatus.UNCERTAIN

        if kind == "comparison":
            return self._evaluate_comparison(
                constraint, claim, completed_verifications,
                evidence_graph_result, order_record,
            )

        return ConstraintStatus.UNCERTAIN

    def _evaluate_comparison(
        self,
        comparison: Dict[str, Any],
        claim: ClaimExtraction,
        completed_verifications: List[str],
        evidence_graph_result: EvidenceGraphSearchResult,
        order_record: Optional[Dict[str, Any]],
    ) -> ConstraintStatus:
        left = self._operand_value(
            comparison.get("left"), claim, completed_verifications,
            evidence_graph_result, order_record,
        )
        right = self._operand_value(
            comparison.get("right"), claim, completed_verifications,
            evidence_graph_result, order_record,
        )
        if left is None or right is None:
            return ConstraintStatus.UNCERTAIN

        operator = comparison.get("operator")
        try:
            result = {
                "==": lambda: left == right,
                "!=": lambda: left != right,
                ">": lambda: left > right,
                ">=": lambda: left >= right,
                "<": lambda: left < right,
                "<=": lambda: left <= right,
                "in": lambda: left in right,
                "not_in": lambda: left not in right,
            }.get(operator, lambda: None)()
        except (TypeError, ValueError):
            return ConstraintStatus.UNCERTAIN
        if result is None:
            return ConstraintStatus.UNCERTAIN
        return (
            ConstraintStatus.SATISFIED if result
            else ConstraintStatus.VIOLATED
        )

    def _operand_value(
        self,
        operand: Any,
        claim: ClaimExtraction,
        completed_verifications: List[str],
        evidence_graph_result: EvidenceGraphSearchResult,
        order_record: Optional[Dict[str, Any]],
    ) -> Any:
        if not isinstance(operand, dict):
            return None
        if "value" in operand:
            return operand["value"]
        field_name = operand.get("field")
        return self._resolve_policy_field(
            field_name, claim, completed_verifications,
            evidence_graph_result, order_record,
        )

    def _resolve_policy_field(
        self,
        field_name: Optional[str],
        claim: ClaimExtraction,
        completed_verifications: List[str],
        evidence_graph_result: EvidenceGraphSearchResult,
        order_record: Optional[Dict[str, Any]],
    ) -> Any:
        if not field_name:
            return None

        evidence_status = evidence_graph_result.status_for_field(field_name)
        evidence_dependent_fields = {
            "order_id",
            "payment_record", "amount_paid", "required_evidence_status",
            "police_report_verified", "adjuster_report_status",
            "signature_captured",
        }
        if (
            field_name in evidence_dependent_fields and
            evidence_status in {"missing", "reported"}
        ):
            return None

        raw_text = claim.raw_text.lower()
        facts: Dict[str, Any] = {
            "order_id": claim.order_id,
            "proposed_refund_amount": claim.amount if claim.amount > 0 else None,
            "order_total": order_record.get("amount") if order_record else None,
            "order_value": (
                order_record.get("amount") if order_record
                else claim.amount if claim.amount > 0 else None
            ),
            "amount_paid": (
                (
                    order_record.get("amount_paid", order_record.get("amount"))
                    if order_record else None
                )
            ),
            "remaining_refundable_amount": (
                order_record.get("remaining_refundable_amount")
                if order_record else None
            ),
            "payment_record": (
                (
                    order_record.get("payment_record", {
                        "order_id": claim.order_id,
                        "amount": order_record.get("amount"),
                    })
                    if order_record else None
                )
            ),
            "order_exists": True if order_record else None,
            "order_status": self._get_order_status(order_record),
            "shipment_status": self._get_order_status(order_record),
            "fulfillment_status": self._get_order_status(order_record),
            "delivery_status": (
                "delivered" if order_record and order_record.get("delivered")
                else "not_delivered" if order_record
                and order_record.get("delivered") is False
                else None
            ),
            "purchase_date": (
                claim.purchase_date or
                (order_record.get("date") if order_record else None)
            ),
            "claim_date": datetime.now().strftime("%Y-%m-%d"),
            "reported_defect": self._reported_defect(claim.raw_text),
            "incident_type": self._incident_type(raw_text),
            "previous_refund_check_status": (
                "completed" if "history_check" in completed_verifications else None
            ),
            "refund_authorization_status": (
                "authorized" if "manager_approval" in completed_verifications else None
            ),
            "refund_prerequisites_status": (
                "satisfied" if "inspection_check" in completed_verifications else None
            ),
            "refund_status_verification": (
                "verified" if "refund_status_check" in completed_verifications else None
            ),
            "police_report_verified": self._verified_evidence_value(
                "police_report", evidence_graph_result, completed_verifications
            ),
            "adjuster_report_status": self._verified_evidence_value(
                "adjuster_report", evidence_graph_result, completed_verifications,
                verified_value="completed",
            ),
            "signature_captured": self._verified_evidence_value(
                "delivery_signature", evidence_graph_result, completed_verifications
            ),
            "required_evidence_status": self._required_evidence_status(
                evidence_graph_result, completed_verifications
            ),
            "item_condition": (
                "inspected" if "inspection_check" in completed_verifications else None
            ),
            "returned_item_received": (
                True if "item_returned" in completed_verifications else None
            ),
            "previous_exchanges_count": (
                self._customer_db["default"].get("previous_exchanges")
            ),
            "refund_status": None,
        }
        return facts.get(field_name)

    @staticmethod
    def _get_order_status(order_record: Optional[Dict[str, Any]]) -> Optional[str]:
        if not order_record:
            return None
        if order_record.get("shipped") is True:
            return "shipped"
        if order_record.get("shipped") is False:
            return "not_shipped"
        return order_record.get("status")

    @staticmethod
    def _reported_defect(raw_text: str) -> Optional[str]:
        for defect in (
            "stopped charging", "doesn't charge", "won't turn on",
            "stopped working", "broken", "cracked", "damaged", "defective",
        ):
            if defect in raw_text.lower():
                return defect
        return None

    @staticmethod
    def _incident_type(raw_text: str) -> Optional[str]:
        text = raw_text.lower()
        if re.search(r"\b(?:theft|stolen)\b", text):
            return "theft"
        if re.search(r"\b(?:lost|loss)\b", text):
            return "loss"
        if re.search(r"\b(?:damage|damaged)\b", text):
            return "damage"
        return None

    @staticmethod
    def _verified_evidence_value(
        evidence_type: str,
        evidence_graph_result: EvidenceGraphSearchResult,
        completed_verifications: List[str],
        verified_value: Any = True,
    ) -> Any:
        status = next(
            (
                item["status"] for item in evidence_graph_result.provided_evidence
                if item["type"] == evidence_type
            ),
            None,
        )
        if status == "verified":
            return verified_value
        if "evidence_check" in completed_verifications:
            return False if verified_value is True else "missing"
        return None

    @staticmethod
    def _required_evidence_status(
        evidence_graph_result: EvidenceGraphSearchResult,
        completed_verifications: List[str],
    ) -> Optional[str]:
        relevant = [
            item for item in evidence_graph_result.provided_evidence
            if item["type"] != "order_id"
        ]
        if any(item["status"] == "verified" for item in relevant):
            return "verified"
        if "evidence_check" in completed_verifications:
            return "missing"
        return None

    def _evaluate_single_constraint(self, constraint_def: Dict,
                                    claim: ClaimExtraction,
                                    completed_verifications: List[str]) -> ConstraintStatus:
        """Evaluate a single constraint against the claim data."""
        variable = constraint_def["variable"]
        ctype = constraint_def.get("type", "")

        # --- Temporal constraints (purchase window) ---
        if ctype == "temporal":
            return self._check_temporal(constraint_def, claim)

        # --- Evidence constraints (receipt, photos) ---
        if ctype == "evidence":
            return self._check_evidence(constraint_def, claim, completed_verifications)

        # --- Financial constraints (amount limits) ---
        if ctype == "financial":
            return self._check_financial(constraint_def, claim)

        # --- History constraints (claim frequency) ---
        if ctype == "history":
            return self._check_history(constraint_def, claim, completed_verifications)

        # --- Condition constraints (item condition) ---
        if ctype == "condition":
            return self._check_condition(constraint_def, claim)

        # --- Status constraints (order status) ---
        if ctype == "status":
            return self._check_status(constraint_def, claim)

        # --- Approval constraints (manager approval) ---
        if ctype == "approval":
            return self._check_approval(constraint_def, claim, completed_verifications)

        # Unknown constraint type — mark uncertain
        return ConstraintStatus.UNCERTAIN

    def _check_temporal(self, constraint_def: Dict,
                        claim: ClaimExtraction) -> ConstraintStatus:
        """Check if the request is within the allowed time window."""
        purchase_date = claim.purchase_date
        if not purchase_date and claim.order_id:
            order = self._customer_db["default"]["order_history"].get(claim.order_id)
            if order:
                purchase_date = order["date"]

        if purchase_date:
            try:
                # Try to parse the date
                for fmt in (
                    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
                ):
                    try:
                        purchase = datetime.strptime(purchase_date, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return ConstraintStatus.UNCERTAIN

                if "window_hours" in constraint_def:
                    allowed_age = timedelta(hours=constraint_def["window_hours"])
                else:
                    allowed_age = timedelta(days=constraint_def.get("window_days", 30))
                now = datetime.now()
                age = now - purchase
                if age < timedelta(0):
                    return ConstraintStatus.POTENTIALLY_MANIPULATED
                if age <= allowed_age:
                    return ConstraintStatus.SATISFIED
                else:
                    return ConstraintStatus.VIOLATED
            except Exception:
                return ConstraintStatus.UNCERTAIN
        else:
            # No date provided — uncertain
            return ConstraintStatus.UNCERTAIN

    def _check_evidence(self, constraint_def: Dict, claim: ClaimExtraction,
                        completed_verifications: List[str]) -> ConstraintStatus:
        """Check if required evidence has been provided."""
        variable = constraint_def["variable"]
        required_if = constraint_def.get("required_if")

        # Conditional evidence requirement
        if required_if:
            # Check if the condition applies
            if required_if == "item_claimed_defective":
                defect_terms = ("defect", "broken", "cracked", "damaged",
                                "stopped working", "stopped charging",
                                "doesn't work", "won't turn on")
                if claim.intent not in ("refund", "warranty") or not any(
                    term in claim.raw_text.lower() for term in defect_terms
                ):
                    return ConstraintStatus.SATISFIED  # Not applicable

        # Check if evidence is in the claim or has been verified
        evidence_map = {
            "receipt_provided": ["receipt", "proof of purchase", "order confirmation",
                                 "invoice", "order_id"],
            "photo_evidence_provided": ["photo", "picture", "image", "screenshot"],
        }

        relevant_evidence = evidence_map.get(variable, [variable])
        claim_evidence_lower = [e.lower() for e in claim.evidence_provided]

        if "evidence_check" in completed_verifications or \
           variable in completed_verifications or \
           f"verify_{variable}" in completed_verifications:
            return ConstraintStatus.SATISFIED

        # Check if any matching evidence was provided
        for ev in relevant_evidence:
            if any(ev in ce for ce in claim_evidence_lower):
                if "fabricated_evidence" in claim.adversarial_tactics:
                    return ConstraintStatus.POTENTIALLY_MANIPULATED
                return ConstraintStatus.UNCERTAIN

        # If order_id is provided, we can look up the receipt
        if variable == "receipt_provided" and claim.order_id and \
           self.get_order_record(claim.order_id):
            return ConstraintStatus.SATISFIED

        return ConstraintStatus.UNCERTAIN

    def _check_financial(self, constraint_def: Dict,
                         claim: ClaimExtraction) -> ConstraintStatus:
        """Check if the claimed amount is within acceptable limits."""
        if claim.amount <= 0:
            return ConstraintStatus.UNCERTAIN

        # Look up actual purchase amount from order history
        customer = self._customer_db.get("default", {})
        orders = customer.get("order_history", {})

        if claim.order_id and claim.order_id in orders:
            actual_amount = orders[claim.order_id]["amount"]
            if claim.amount <= actual_amount:
                return ConstraintStatus.SATISFIED
            else:
                return ConstraintStatus.POTENTIALLY_MANIPULATED

        # A plausible amount is not proof of the original transaction value.
        return ConstraintStatus.UNCERTAIN

    def _check_history(self, constraint_def: Dict,
                       claim: ClaimExtraction,
                       completed_verifications: List[str]) -> ConstraintStatus:
        """Check the customer's claim frequency history."""
        max_claims = constraint_def.get("max_claims", 3)
        customer = self._customer_db.get("default", {})
        previous = customer.get("previous_claims_6mo", 0)

        if claim.claimed_previous_claims is not None:
            if "history_check" not in completed_verifications:
                return ConstraintStatus.UNCERTAIN
            if claim.claimed_previous_claims > previous:
                return ConstraintStatus.POTENTIALLY_MANIPULATED

        if previous < max_claims:
            return ConstraintStatus.SATISFIED
        elif previous == max_claims:
            return ConstraintStatus.UNCERTAIN
        else:
            return ConstraintStatus.VIOLATED

    def _check_condition(self, constraint_def: Dict,
                         claim: ClaimExtraction) -> ConstraintStatus:
        """Check item condition requirements."""
        acceptable = constraint_def.get("acceptable_values", [])

        # Extract condition from claim text
        text_lower = claim.raw_text.lower()
        condition_keywords = {
            "original": ["original", "unopened", "sealed", "new"],
            "defective": ["defective", "broken", "doesn't work", "malfunctioning",
                          "stopped working", "stopped charging", "doesn't charge",
                          "dead", "cracked"],
            "damaged_in_transit": ["damaged in shipping", "arrived broken",
                                   "damaged in transit", "delivery damage",
                                   "cracked during shipping", "cracked during delivery"],
            "unopened": ["unopened", "sealed", "never opened"],
            "used": ["used", "worn", "opened"],
        }

        detected_condition = None
        for condition, keywords in condition_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected_condition = condition
                break

        if detected_condition is None:
            return ConstraintStatus.UNCERTAIN

        if detected_condition in acceptable:
            return ConstraintStatus.SATISFIED

        return ConstraintStatus.VIOLATED

    def _check_status(self, constraint_def: Dict,
                      claim: ClaimExtraction) -> ConstraintStatus:
        """Check order status constraints."""
        required_status = constraint_def.get("required_status", "")
        customer = self._customer_db.get("default", {})
        orders = customer.get("order_history", {})

        if claim.order_id and claim.order_id in orders:
            order = orders[claim.order_id]
            if required_status == "pre_shipment":
                if not order.get("shipped", False):
                    return ConstraintStatus.SATISFIED
                else:
                    return ConstraintStatus.VIOLATED
        return ConstraintStatus.UNCERTAIN

    def _check_approval(self, constraint_def: Dict, claim: ClaimExtraction,
                        completed_verifications: List[str]) -> ConstraintStatus:
        """Check if required approvals have been obtained."""
        threshold = constraint_def.get("threshold", 200)

        if claim.amount <= threshold:
            return ConstraintStatus.SATISFIED  # Approval not needed

        if "manager_approval" in completed_verifications:
            return ConstraintStatus.SATISFIED

        return ConstraintStatus.UNCERTAIN

    def _build_explanation(self, constraints: Dict[str, ConstraintStatus],
                           policy_ids: List[str]) -> str:
        """Build a human-readable explanation of the CSP evaluation."""
        satisfied = sum(1 for s in constraints.values()
                        if s == ConstraintStatus.SATISFIED)
        violated = sum(1 for s in constraints.values()
                       if s == ConstraintStatus.VIOLATED)
        uncertain = sum(1 for s in constraints.values()
                        if s == ConstraintStatus.UNCERTAIN)
        manipulated = sum(1 for s in constraints.values()
                          if s == ConstraintStatus.POTENTIALLY_MANIPULATED)

        total = len(constraints)
        parts = [
            f"Evaluated {total} constraints across policies {', '.join(policy_ids)}.",
            f"Results: {satisfied} satisfied, {violated} violated, "
            f"{uncertain} uncertain, {manipulated} potentially manipulated."
        ]

        if violated > 0:
            violated_names = [k for k, v in constraints.items()
                              if v == ConstraintStatus.VIOLATED]
            parts.append(f"Violations: {', '.join(violated_names)}.")

        if manipulated > 0:
            parts.append("⚠ Potential manipulation detected — triggering risk assessment.")
        elif uncertain > 0:
            parts.append("Some constraints need verification — triggering uncertainty analysis.")

        return " ".join(parts)
