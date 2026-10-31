"""Evidence graph connecting user-reported documents to policy constraints."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from shared.models import ClaimExtraction

from .policy_graph import Policy, PolicyGraph


@dataclass
class EvidenceGraphSearchResult:
    policy_ids: List[str]
    required_evidence: List[Dict[str, str]]
    provided_evidence: List[Dict[str, str]]
    links: List[Dict[str, str]]
    field_statuses: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applicable_policy_ids": self.policy_ids,
            "required_evidence": [
                {
                    **requirement,
                    "status": self.field_statuses.get(requirement["field"], "missing"),
                }
                for requirement in self.required_evidence
            ],
            "provided_evidence": self.provided_evidence,
            "evidence_links": self.links,
        }

    def status_for_field(self, field: str) -> Optional[str]:
        return self.field_statuses.get(field)


class EvidenceGraph:
    """Indexes policy evidence requirements and links them to a claim's evidence."""

    FIELD_EVIDENCE: Dict[str, Dict[str, Any]] = {
        "order_id": {
            "label": "Order identifier and transaction record",
            "types": ["order_id"],
        },
        "payment_record": {
            "label": "Payment or purchase transaction record",
            "types": [
                "transaction_record", "receipt", "payment_confirmation",
                "bank_statement",
            ],
        },
        "amount_paid": {
            "label": "Original amount paid from the transaction record",
            "types": [
                "transaction_record", "receipt", "payment_confirmation",
                "bank_statement",
            ],
        },
        "return_cancellation_status": {
            "label": "Verified return or cancellation outcome",
            "types": ["return_confirmation", "cancellation_confirmation"],
        },
        "previous_refund_check_status": {
            "label": "Previous refund history check",
            "types": ["refund_history"],
        },
        "refund_prerequisites_status": {
            "label": "Return receipt and required inspection checks",
            "types": ["return_confirmation", "inspection_report"],
        },
        "refund_status_verification": {
            "label": "Refund status retrieved from the payment or order system",
            "types": ["refund_confirmation"],
        },
        "required_evidence_status": {
            "label": "Evidence required by the warranty terms",
            "types": [
                "receipt", "photo", "video", "warranty_card",
                "inspection_report", "other_document",
            ],
        },
        "police_report_verified": {
            "label": "Official police report",
            "types": ["police_report"],
        },
        "adjuster_report_status": {
            "label": "Official adjuster assessment",
            "types": ["adjuster_report"],
        },
        "signature_captured": {
            "label": "Proof of delivery signature",
            "types": ["delivery_signature"],
        },
        "item_condition": {
            "label": "Returned-item condition inspection",
            "types": ["inspection_report"],
        },
    }

    def __init__(self, policy_graph: PolicyGraph):
        self.policy_graph = policy_graph
        self.requirements_by_policy: Dict[str, List[Dict[str, str]]] = {
            policy.policy_id: self._build_requirements(policy)
            for policy in policy_graph.get_all_policies()
        }

    def search(self, claim: ClaimExtraction,
               completed_verifications: Optional[List[str]] = None,
               order_record: Optional[Dict[str, Any]] = None
               ) -> EvidenceGraphSearchResult:
        """Search the relevant policy subgraph and link evidence mentioned by NLU."""
        policies = self.policy_graph.search_applicable_policies(claim.intent)
        verifications = set(completed_verifications or [])
        provided = set(claim.evidence_provided)
        if claim.order_id:
            provided.add("order_id")
        if order_record:
            provided.add("transaction_record")

        requirements = [
            requirement
            for policy in policies
            for requirement in self.requirements_by_policy.get(policy.policy_id, [])
        ]
        field_statuses: Dict[str, str] = {}
        provided_nodes = [
            {
                "type": evidence_type,
                "status": (
                    "verified" if (
                        evidence_type in {"order_id", "transaction_record"} and
                        order_record
                    ) or (
                        "evidence_check" in verifications and
                        evidence_type != "order_id"
                    ) else "reported"
                ),
            }
            for evidence_type in sorted(provided)
        ]

        links: List[Dict[str, str]] = []
        for requirement in requirements:
            field = requirement["field"]
            matching_types = [
                evidence_type for evidence_type in requirement["evidence_types"].split(",")
                if evidence_type in provided
            ]
            if field == "order_id":
                status = "verified" if order_record else "missing"
            elif matching_types:
                status = (
                    "verified" if (
                        "transaction_record" in matching_types and order_record
                    ) or (
                        "evidence_check" in verifications
                    ) else "reported"
                )
            else:
                status = "missing"
            field_statuses[field] = status
            for evidence_type in matching_types:
                links.append({
                    "evidence_type": evidence_type,
                    "constraint_id": requirement["constraint_id"],
                    "field": field,
                    "status": status,
                })

        return EvidenceGraphSearchResult(
            policy_ids=[policy.policy_id for policy in policies],
            required_evidence=requirements,
            provided_evidence=provided_nodes,
            links=links,
            field_statuses=field_statuses,
        )

    def _build_requirements(self, policy: Policy) -> List[Dict[str, str]]:
        requirements: List[Dict[str, str]] = []
        for constraint in policy.get_all_constraints():
            description = constraint.get("description", "")
            for field in sorted(self._collect_fields(constraint)):
                metadata = self.FIELD_EVIDENCE.get(field)
                field_is_evidence = re.search(
                    r"evidence|proof|receipt|report|signature|inspection|document",
                    field,
                    re.IGNORECASE,
                )
                if metadata is None and not field_is_evidence:
                    continue
                metadata = metadata or {
                    "label": description or field.replace("_", " "),
                    "types": ["other_document"],
                }
                requirements.append({
                    "policy_id": policy.policy_id,
                    "constraint_id": constraint["_clause_id"],
                    "field": field,
                    "label": metadata["label"],
                    "evidence_types": ",".join(metadata["types"]),
                })
        return requirements

    @classmethod
    def _collect_fields(cls, value: Any) -> Set[str]:
        fields: Set[str] = set()
        if isinstance(value, dict):
            field = value.get("field")
            if isinstance(field, str):
                fields.add(field)
            for child in value.values():
                fields.update(cls._collect_fields(child))
        elif isinstance(value, list):
            for child in value:
                fields.update(cls._collect_fields(child))
        return fields
