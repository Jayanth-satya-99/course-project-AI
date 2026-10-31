# Module 1 — Policy Graph
# =======================
# Defines organisational policies as structured graph data.
# Each policy has clauses with specific constraints that must be evaluated
# against customer claims.

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class PolicyClause:
    """A single clause within a policy, containing the constraints that must hold."""

    def __init__(self, clause_id: str, description: str, constraints: List[Dict[str, Any]],
                 priority: int = 1):
        self.clause_id = clause_id
        self.description = description
        self.constraints = constraints  # list of constraint definitions
        self.priority = priority        # higher = more critical

    def __repr__(self):
        return f"PolicyClause({self.clause_id}: {self.description})"


class Policy:
    """A complete policy with multiple clauses."""

    def __init__(self, policy_id: str, name: str, description: str,
                 clauses: List[PolicyClause], policy_type: str = "",
                 possible_actions: Optional[List[str]] = None):
        self.policy_id = policy_id
        self.name = name
        self.description = description
        self.policy_type = policy_type
        self.possible_actions = possible_actions or []
        self.clauses = {c.clause_id: c for c in clauses}

    def get_clause(self, clause_id: str) -> PolicyClause:
        return self.clauses.get(clause_id)

    def get_all_constraints(self) -> List[Dict[str, Any]]:
        all_constraints = []
        for clause in self.clauses.values():
            for constraint in clause.constraints:
                all_constraints.append({**constraint, "_clause_id": clause.clause_id})
        return all_constraints


class PolicyGraph:
    """
    The organisational policy knowledge base.

    Contains all policies as a searchable graph. Module 1's CSP solver
    queries this graph to find applicable policies and their constraints
    for a given customer request.
    """

    def __init__(self, source_path: Optional[str] = None):
        self.source_path = Path(source_path) if source_path else (
            Path(__file__).resolve().parent.parent / "policies.json"
        )
        self.metadata: Dict[str, Any] = {}
        self.evaluation_statuses: List[str] = []
        self.policies: Dict[str, Policy] = {}
        self._load_policies()

    def _load_policies(self) -> None:
        """Build the searchable policy graph from the project JSON policy source."""
        data = json.loads(self.source_path.read_text(encoding="utf-8"))
        self.metadata = {
            key: value for key, value in data.items()
            if key not in {"policies", "evaluation_statuses"}
        }
        self.evaluation_statuses = data.get("evaluation_statuses", [])
        raw_policies = data.get("policies")
        if not isinstance(raw_policies, list) or not raw_policies:
            raise ValueError(f"No policies were found in {self.source_path}.")

        for raw_policy in raw_policies:
            policy_id = raw_policy.get("policy_id")
            policy_type = raw_policy.get("type")
            constraints = raw_policy.get("constraints")
            if not all((policy_id, policy_type, isinstance(constraints, list))):
                raise ValueError(
                    f"Policy entry in {self.source_path} is missing its id, type, "
                    "or constraints list."
                )

            clauses = []
            for constraint in constraints:
                if not isinstance(constraint, dict):
                    raise ValueError(
                        f"Invalid constraint in policy {policy_id}: "
                        "every constraint needs a constraint_id."
                    )
                constraint_id = constraint.get("constraint_id")
                if not constraint_id:
                    raise ValueError(
                        f"Invalid constraint in policy {policy_id}: "
                        "every constraint needs a constraint_id."
                    )
                clauses.append(PolicyClause(
                    clause_id=constraint_id,
                    description=constraint.get("description", constraint_id),
                    constraints=[constraint],
                ))

            policy = Policy(
                policy_id=policy_id,
                name=raw_policy.get("name", policy_id),
                description=raw_policy.get("description", ""),
                clauses=clauses,
                policy_type=policy_type,
                possible_actions=raw_policy.get("possible_actions", []),
            )
            if policy_type in self.policies:
                raise ValueError(f"Duplicate policy type in {self.source_path}: {policy_type}")
            self.policies[policy_type] = policy

    def search_applicable_policies(self, intent: str) -> List[Policy]:
        """
        Search for policies whose JSON policy type matches the normalized intent.
        """
        intent_lower = intent.lower().strip()
        aliases = {
            "cancel": "cancellation",
            "refund_request": "refund",
            "return_request": "return",
            "modification": "order_modification",
            "modify_order": "order_modification",
        }
        policy_type = aliases.get(intent_lower, intent_lower)
        policy = self.policies.get(policy_type)
        return [policy] if policy else []

    def search_relevant_constraints(self, policies: List[Policy]) -> List[Dict[str, Any]]:
        """Return policy clauses as searchable CSP constraint records."""
        results = []
        for policy in policies:
            for constraint in policy.get_all_constraints():
                results.append({
                    **constraint,
                    "_policy_id": policy.policy_id,
                    "_policy_type": policy.policy_type,
                })
        return results

    def get_policy_by_id(self, policy_id: str) -> Policy:
        """Retrieve a specific policy by its ID."""
        for policy in self.policies.values():
            if policy.policy_id == policy_id:
                return policy
        return None

    def get_all_policies(self) -> List[Policy]:
        """Return all unique policies."""
        seen = set()
        unique = []
        for policy in self.policies.values():
            if policy.policy_id not in seen:
                seen.add(policy.policy_id)
                unique.append(policy)
        return unique
