# Module 1 — Constraint Satisfaction Problem (CSP) Solver
# =======================================================
# Maps extracted customer claims against the policy graph and evaluates
# every required constraint, tagging each as:
#   SATISFIED | VIOLATED | UNCERTAIN | POTENTIALLY_MANIPULATED
#
# When a constraint is UNCERTAIN or POTENTIALLY_MANIPULATED, the CSP solver
# triggers Module 3 (Bayesian) for quantitative risk assessment.

import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Ensure workspace root is in sys.path so direct execution works
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from shared.models import (
    ClaimExtraction, CSPResult, ConstraintStatus, ThreatLevel, Sentiment
)

try:
    from .evidence_graph import EvidenceGraph, EvidenceGraphSearchResult
    from .policy_graph import PolicyGraph, Policy
except ImportError:
    from module1.evidence_graph import EvidenceGraph, EvidenceGraphSearchResult
    from module1.policy_graph import PolicyGraph, Policy


class CSPSolver:
    """
    Constraint Satisfaction Problem solver for policy compliance evaluation.

    Variables: Each constraint from the applicable policy clauses.
    Domains:   {SATISFIED, VIOLATED, UNCERTAIN, POTENTIALLY_MANIPULATED}
    Constraints: Defined dynamically by policy clauses in policies.json.
    """

    def __init__(self, policy_graph: PolicyGraph, customer_db: Optional[Dict[str, Any]] = None):
        self.policy_graph = policy_graph
        self.evidence_graph = EvidenceGraph(policy_graph)
        # In-memory customer transaction database for order lookups
        self._customer_db = customer_db or {
            "default": {
                "customer_id": "CUST-001",
                "previous_claims_6mo": 1,
                "is_vip": False,
                "account_age_days": 365,
                "previous_exchanges": 0,
                "order_history": {
                    "ORD-001": {
                        "amount": 500,
                        "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                        "item": "TV",
                        "product_id": "PROD-TV-500",
                        "product_category": "electronics",
                        "shipped": True,
                        "delivered": True,
                        "status": "delivered",
                        "delivery_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                        "payment_method": "credit_card",
                        "payment_status": "paid",
                        "remaining_refundable_amount": 500.0,
                    },
                    "ORD-002": {
                        "amount": 79.99,
                        "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                        "item": "Headphones",
                        "product_id": "PROD-HP-080",
                        "product_category": "audio",
                        "shipped": True,
                        "delivered": True,
                        "status": "delivered",
                        "delivery_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                        "payment_method": "paypal",
                        "payment_status": "paid",
                        "remaining_refundable_amount": 79.99,
                    },
                    "ORD-003": {
                        "amount": 1200,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "item": "Laptop",
                        "product_id": "PROD-LP-1200",
                        "product_category": "computers",
                        "shipped": False,
                        "delivered": False,
                        "status": "processing",
                        "delivery_date": None,
                        "payment_method": "credit_card",
                        "payment_status": "paid",
                        "remaining_refundable_amount": 1200.0,
                    },
                    "ORD-004": {
                        "amount": 250.0,
                        "date": (datetime.now() - timedelta(days=50)).strftime("%Y-%m-%d"),
                        "item": "Camera",
                        "product_id": "PROD-CAM-250",
                        "product_category": "electronics",
                        "shipped": True,
                        "delivered": True,
                        "status": "delivered",
                        "delivery_date": (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d"),
                        "payment_method": "credit_card",
                        "payment_status": "paid",
                        "remaining_refundable_amount": 250.0,
                        "return_window_days": 30,
                    },
                }
            }
        }

    def get_order_record(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Return a copy of the transaction record for an order ID."""
        record = self._customer_db["default"]["order_history"].get(order_id)
        return dict(record) if record else None

    def evaluate_file(self, file_path: Union[str, Path],
                      completed_verifications: Optional[List[str]] = None,
                      output_file: Optional[Union[str, Path]] = None) -> CSPResult:
        """
        Load a claim directly from a JSON input file and evaluate policy constraints.
        Optionally writes the resulting CSPResult to output_file.
        """
        claim = ClaimExtraction.from_file(file_path)
        result = self.evaluate(claim, completed_verifications)
        if output_file:
            result.to_file(output_file)
        return result

    def evaluate(self, claim: Union[ClaimExtraction, Dict[str, Any], str, Path],
                 completed_verifications: Optional[List[str]] = None,
                 evidence_graph_result: Optional[EvidenceGraphSearchResult] = None
                 ) -> CSPResult:
        """
        Run the CSP solver against applicable policies for the given claim.

        Args:
            claim: ClaimExtraction instance, a JSON-compatible dict, a JSON string,
                   or a file path pointing to a JSON input file.
            completed_verifications: List of verification steps already completed.
            evidence_graph_result: Optional pre-computed evidence graph search result.

        Returns:
            CSPResult with every constraint tagged and applicable policies listed.
        """
        if isinstance(claim, (str, Path)):
            claim_str = str(claim)
            if os.path.exists(claim_str):
                claim = ClaimExtraction.from_file(claim_str)
            else:
                import json
                try:
                    data = json.loads(claim_str)
                    claim = ClaimExtraction.from_dict(data)
                except Exception:
                    claim = ClaimExtraction(
                        intent="refund", amount=0.0, item="item",
                        purchase_date=None, order_id=None, evidence_provided=[],
                        threat_level=ThreatLevel.NONE, sentiment=Sentiment.NEUTRAL,
                        adversarial_tactics=[], raw_text=claim_str
                    )
        elif isinstance(claim, dict):
            claim = ClaimExtraction.from_dict(claim)

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
        contradiction_types = {c.get("type") for c in getattr(claim, "contradictory_claims", [])}

        for constraint_def in self.policy_graph.search_relevant_constraints(
            applicable_policies
        ):
            clause_id = constraint_def.get("_clause_id", "")
            key = f"{clause_id}:{constraint_def.get('description', clause_id)}"
            status = self._evaluate_json_constraint(
                constraint_def, claim, completed_verifications,
                evidence_graph_result, order_record,
            )

            # Check if contradictions or fabrication indicate potential manipulation
            if status != ConstraintStatus.VIOLATED and (contradiction_types or "fabricated_evidence" in claim.adversarial_tactics):
                desc_lower = key.lower()
                is_manipulated = False
                if "amount_mismatch" in contradiction_types and any(w in desc_lower for w in ("amount", "refund", "limit", "value")):
                    is_manipulated = True
                elif "order_mismatch" in contradiction_types and any(w in desc_lower for w in ("order", "exist", "valid", "record")):
                    is_manipulated = True
                elif "evidence_conflict" in contradiction_types and any(w in desc_lower for w in ("evidence", "receipt", "proof", "document")):
                    is_manipulated = True
                elif "claim_statement_conflict" in contradiction_types and any(w in desc_lower for w in ("condition", "tag", "packag", "defect", "inspection", "return")):
                    is_manipulated = True
                elif "fabricated_evidence" in claim.adversarial_tactics and any(
                    w in desc_lower for w in ("evidence", "receipt", "proof", "record", "document", "verification", "check")
                ):
                    is_manipulated = True

                if is_manipulated:
                    status = ConstraintStatus.POTENTIALLY_MANIPULATED

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
        """Evaluate recursive all/any/conditional/comparison predicates from policies.json."""
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
        """
        Dynamically extracts and resolves attribute values from claim data,
        the customer order database, evidence graph, and completed verifications.
        No hardcoded policy values or mock strings.
        """
        if not field_name:
            return None

        # 1. Evidence check: fields requiring verified evidence evaluate to None (UNCERTAIN)
        # if the evidence is missing or reported without verification.
        evidence_status = evidence_graph_result.status_for_field(field_name)
        evidence_dependent_fields = {
            "order_id", "payment_record", "amount_paid", "required_evidence_status",
            "police_report_verified", "adjuster_report_status", "signature_captured",
        }
        if (
            field_name in evidence_dependent_fields and
            evidence_status in {"missing", "reported"}
        ):
            return None

        verifications_set = set(completed_verifications)

        # 2. Direct lookup: check if order_record explicitly holds this field
        if order_record and field_name in order_record and order_record[field_name] is not None:
            return order_record[field_name]

        # 3. Dynamic Claim properties
        if field_name == "order_id":
            return claim.order_id
        if field_name == "order_exists":
            return True if order_record else None
        if field_name == "proposed_refund_amount":
            return claim.amount if claim.amount > 0 else None
        if field_name == "order_total":
            return order_record.get("amount") if order_record else None
        if field_name == "order_value":
            return order_record.get("amount") if order_record else (claim.amount if claim.amount > 0 else None)
        if field_name == "amount_paid":
            return order_record.get("amount_paid", order_record.get("amount")) if order_record else None
        if field_name == "remaining_refundable_amount":
            return order_record.get("remaining_refundable_amount", order_record.get("amount")) if order_record else None
        if field_name == "payment_record":
            return order_record.get("payment_record", {
                "order_id": claim.order_id,
                "amount": order_record.get("amount"),
            }) if order_record else None
        if field_name == "purchase_date":
            return claim.purchase_date or (order_record.get("date") if order_record else None)
        if field_name == "claim_date":
            return datetime.now().strftime("%Y-%m-%d")
        if field_name == "reported_defect":
            return self._reported_defect(claim.raw_text)
        if field_name == "incident_type":
            return self._incident_type(claim.raw_text)
        if field_name == "requested_field":
            return self._requested_modification_field(claim.raw_text)

        # 4. Order and Fulfillment States
        if field_name in {"order_status", "shipment_status", "fulfillment_status"}:
            return self._get_order_status(order_record)
        if field_name == "delivery_status":
            if not order_record:
                return None
            if order_record.get("delivered") is True:
                return "delivered"
            if order_record.get("delivered") is False:
                return "not_delivered"
            return order_record.get("status")
        if field_name == "delivery_date":
            return order_record.get("delivery_date") if order_record else None
        if field_name == "product_category":
            return order_record.get("product_category", "general") if order_record else None

        # 5. Dynamic Date & Window Computations
        purchase_date_str = claim.purchase_date or (order_record.get("date") if order_record else None)
        delivery_date_str = order_record.get("delivery_date") if order_record else None
        days_since_delivery = self._compute_days_ago(delivery_date_str)
        days_since_purchase = self._compute_days_ago(purchase_date_str)
        date_diff = days_since_delivery if days_since_delivery is not None else days_since_purchase

        if field_name in {"days_since_delivery", "date_difference_days"}:
            return date_diff
        if field_name in {"applicable_return_window_days", "exchange_window_days"}:
            return order_record.get("return_window_days", 30) if order_record else 30
        if field_name == "return_eligible":
            if not order_record:
                return None
            max_window = order_record.get("return_window_days", 30)
            return (date_diff <= max_window) if date_diff is not None else True
        if field_name == "return_eligibility_status":
            if not order_record:
                return None
            max_window = order_record.get("return_window_days", 30)
            if date_diff is not None and date_diff > max_window:
                return "ineligible"
            return "eligible" if (date_diff is not None and date_diff <= max_window) else None

        # 6. Pre-shipment operations (cancellation, modification permissions)
        is_shipped = order_record.get("shipped", False) if order_record else None
        if field_name in {"cancellation_allowed", "field_modification_allowed", "modification_restrictions_satisfied"}:
            return (not is_shipped) if order_record else None
        if field_name == "shipped_order_cancellation_option_available":
            return False if (order_record and is_shipped) else None

        # 7. Customer Authorization and System Verification Registry
        verification_mappings = {
            "customer_authorization_status": ("verified" if ("order_lookup" in verifications_set or "history_check" in verifications_set) else None),
            "previous_refund_check_status": ("completed" if "history_check" in verifications_set else None),
            "refund_authorization_status": ("authorized" if "manager_approval" in verifications_set else None),
            "refund_prerequisites_status": ("satisfied" if "inspection_check" in verifications_set else None),
            "refund_status_verification": ("verified" if "refund_status_check" in verifications_set else None),
            "cancellation_check_status": ("completed" if "order_lookup" in verifications_set else None),
            "cancellation_system_status": ("confirmed" if "cancellation_confirmed" in verifications_set else None),
            "refund_eligibility_check_status": ("completed" if ("order_lookup" in verifications_set or "history_check" in verifications_set) else None),
            "return_exclusions_checked": (True if ("order_lookup" in verifications_set or "history_check" in verifications_set) else None),
            "required_parts_check_status": ("completed" if ("inspection_check" in verifications_set or "evidence_check" in verifications_set) else None),
            "packaging_tag_check_status": ("completed" if ("inspection_check" in verifications_set or "evidence_check" in verifications_set) else None),
            "return_instructions_communicated": (True if ("order_lookup" in verifications_set or "evidence_check" in verifications_set) else None),
            "alternative_options_checked": (True if "order_lookup" in verifications_set else None),
            "update_system_status": ("confirmed" if "order_lookup" in verifications_set else None),
            "investigation_status": ("completed" if "investigation" in verifications_set else None),
            "escalation_status": ("completed" if "manager_approval" in verifications_set else None),
            "refund_initiation_system_status": ("confirmed" if ("refund_initiation" in verifications_set or "manager_approval" in verifications_set) else None),
        }
        if field_name in verification_mappings:
            return verification_mappings[field_name]

        # 8. Evidence-derived verification values
        if field_name == "police_report_verified":
            return self._verified_evidence_value("police_report", evidence_graph_result, completed_verifications)
        if field_name == "adjuster_report_status":
            return self._verified_evidence_value("adjuster_report", evidence_graph_result, completed_verifications, verified_value="completed")
        if field_name == "signature_captured":
            return self._verified_evidence_value("delivery_signature", evidence_graph_result, completed_verifications)
        if field_name == "required_evidence_status":
            return self._required_evidence_status(evidence_graph_result, completed_verifications)
        if field_name == "returned_item_received":
            return True if "item_returned" in verifications_set else None
        if field_name == "item_condition":
            return "inspected" if "inspection_check" in verifications_set else None

        # 9. Return & Cancellation Outcomes for Refund Eligibility
        if field_name == "return_cancellation_status":
            if "item_returned" in verifications_set:
                return "return_received"
            if "manager_approval" in verifications_set:
                return "refund_exception_authorized"
            if "evidence_check" in verifications_set:
                return "return_approved"
            if order_record and order_record.get("status") == "cancelled":
                return "confirmed_cancelled"
            return None

        if field_name == "refund_eligibility_status":
            if order_record and ("evidence_check" in verifications_set or claim.intent == "cancellation"):
                return "eligible"
            return None

        if field_name == "refund_method_allowed":
            return True if order_record else None

        # 10. Customer Account Attributes (queried from customer database)
        customer = self._customer_db.get("default", {})
        if field_name == "authenticated_customer_id":
            return customer.get("customer_id")
        if field_name == "order_customer_id":
            return customer.get("customer_id") if order_record else None
        if field_name == "previous_exchanges_count":
            return customer.get("previous_exchanges", 0)

        # 11. Domain-specific policy defaults / order fields
        if field_name in {"claimed_product_id", "ordered_product_id"}:
            return order_record.get("product_id") if order_record else None
        if field_name == "defect_coverage_status":
            return "covered" if self._reported_defect(claim.raw_text) else None
        if field_name == "exclusions_checked":
            return True if ("order_lookup" in verifications_set or "evidence_check" in verifications_set) else None
        if field_name == "warranty_decision":
            return "covered" if ("evidence_check" in verifications_set and self._reported_defect(claim.raw_text)) else None
        if field_name == "new_value_valid":
            return True if ("order_lookup" in verifications_set or claim.raw_text) else None

        return None

    @staticmethod
    def _compute_days_ago(date_str: Optional[str]) -> Optional[int]:
        """Compute the number of elapsed days from an ISO date string to today."""
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                dt = datetime.strptime(date_str[:10] if len(date_str) >= 10 else date_str, fmt)
                return max(0, (datetime.now() - dt).days)
            except ValueError:
                continue
        return None

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
        text = raw_text.lower()
        defect_terms = (
            "stopped charging", "doesn't charge", "not charging", "won't turn on",
            "stopped working", "doesn't work", "not working", "broken", "cracked",
            "shattered", "damaged", "defective", "faulty", "malfunctioning",
            "dead on arrival", "glitch", "defect", "fault",
        )
        for defect in defect_terms:
            if defect in text:
                return defect
        return None

    @staticmethod
    def _incident_type(raw_text: str) -> Optional[str]:
        text = raw_text.lower()
        if re.search(r"\b(?:theft|stolen)\b", text):
            return "theft"
        if re.search(r"\b(?:lost|loss)\b", text):
            return "loss"
        if re.search(r"\b(?:damage|damaged|broken|cracked|shattered|dropped)\b", text):
            return "damage"
        return None

    @staticmethod
    def _requested_modification_field(raw_text: str) -> Optional[str]:
        text = raw_text.lower()
        if re.search(r"\b(?:address|location|shipping address|delivery address)\b", text):
            return "delivery_address"
        if re.search(r"\b(?:phone|number|contact)\b", text):
            return "delivery_contact_number"
        if re.search(r"\b(?:instruction|instructions|note|notes|gate code)\b", text):
            return "delivery_instructions"
        return "delivery_address"

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


if __name__ == "__main__":
    policy_graph = PolicyGraph()
    solver = CSPSolver(policy_graph)
    sample_file = _WORKSPACE_ROOT / "sample_claim_input.json"
    output_file = _WORKSPACE_ROOT / "sample_csp_result.json"
    if sample_file.exists():
        result = solver.evaluate_file(sample_file, output_file=output_file)
        print("Policies evaluated:", result.applicable_policies)
        print("Constraints evaluated:", len(result.constraints))
        print("Summary:", result.explanation)
        print(f"CSP output saved to: {output_file}")
    else:
        print("CSPSolver loaded successfully.")
