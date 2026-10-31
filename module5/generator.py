# Module 5 — Response Generator (The Voice)
# ==========================================
# Renders strict logical decisions from Modules 1, 2, 3, and 4 into natural,
# empathetic, and fully transparent customer dialogue.
#
# Policy Requirement:
#   "The final response must clearly explain the decision, the policy clauses
#    considered, the evidence that was accepted or questioned, and any additional
#    action available to the customer."
#
# Grounded Generation Principles:
#   1. Empathy & De-escalation (acknowledges customer frustration without yielding policy).
#   2. Policy Grounding (cites the clauses loaded for the current claim).
#   3. Evidence Transparency (details what was accepted vs what is still missing).
#   4. Actionable Guidance (gives unambiguous next steps for resolution).

import json
import os
import re
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from shared.models import (
    ActionType, ClaimExtraction, CSPResult, RiskAssessment,
    PlanResult, MinimaxResult, GeneratedResponse, Sentiment, ThreatLevel,
    ConstraintStatus,
)


class OllamaGenerationError(RuntimeError):
    """Raised when the configured local model cannot safely generate text."""


class ResponseGenerator:
    """
    Policy-grounded response generator. Ollama, when enabled, writes only the
    empathy opening; decisions and policy details remain deterministic.
    """

    def __init__(self, provider: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: Optional[float] = None):
        self.provider = (provider or os.getenv("GENERATOR_PROVIDER", "ollama")).lower()
        if self.provider not in {"ollama", "template"}:
            raise ValueError(f"Unsupported response generator provider: {self.provider}")

        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("OLLAMA_BASE_URL must be an absolute HTTP(S) URL.")

        self.timeout = (
            float(timeout) if timeout is not None
            else float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "20"))
        )
        if self.timeout <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than zero.")

    def generate(self, action: ActionType,
                 claim: ClaimExtraction,
                 csp_result: Optional[CSPResult],
                 risk_assessment: Optional[RiskAssessment],
                 plan_result: Optional[PlanResult],
                 minimax_result: Optional[MinimaxResult]) -> GeneratedResponse:
        """
        Synthesizes all pipeline outputs into a comprehensive, grounded customer response.
        """
        tone = minimax_result.selected_response_type if minimax_result else "firm_polite"

        # 1. Opening Empathy / De-escalation Hook
        opening = self._generate_opening(claim, tone)
        if self.provider == "ollama":
            opening = self._generate_ollama_opening(tone, claim.sentiment.value)

        # 2. Decision Announcement
        decision_text, cited_clauses = self._generate_decision(action, claim, csp_result, plan_result)

        # 3. Evidence Assessment Summary
        evidence_status = self._generate_evidence_summary(claim, csp_result, risk_assessment)

        # 4. Next Available Actions for the customer
        available_actions = self._generate_available_actions(action, claim, csp_result, plan_result)

        # 5. Assemble Grounded Natural Language Message
        parts = [opening, decision_text, evidence_status]

        if available_actions:
            action_list_str = "\n".join([f"  • {act}" for act in available_actions])
            parts.append(f"\nNext Steps Available to You:\n{action_list_str}")

        closing = self._generate_closing(tone)
        parts.append(closing)

        full_text = "\n\n".join(parts)

        return GeneratedResponse(
            text=full_text,
            cited_policies=cited_clauses,
            evidence_status=evidence_status,
            available_actions=available_actions,
            tone=tone
        )

    def _generate_ollama_opening(self, tone: str, sentiment: str) -> str:
        """Generate a short acknowledgment without sharing claim details."""
        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Write one brief, empathetic acknowledgment for a customer-service "
                        "conversation. Do not mention decisions, refunds, approval, rejection, "
                        "policy, evidence, money, timing, promises, or next steps. Do not add "
                        "facts. Return only the acknowledgment sentence."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Use a {tone} tone. Detected sentiment: {sentiment}.",
                },
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 60},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaGenerationError(
                f"Could not generate an opening with Ollama at {self.base_url}: {exc}"
            ) from exc

        message = payload.get("message") if isinstance(payload, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise OllamaGenerationError(
                "Ollama returned an invalid response: expected message.content text."
            )

        text = text.strip().strip("\"'")
        if not 12 <= len(text) <= 240 or re.search(
            r"\b(?:approve[sd]?|reject(?:ed)?|refund|policy|evidence|"
            r"payment|money|promise|guarantee|escalat(?:e|ed|ion))\b",
            text,
            re.IGNORECASE,
        ):
            raise OllamaGenerationError(
                "Ollama returned an opening that violates the response safety constraints."
            )
        return text

    # ---------------------------------------------------------------------------
    # Generation Subcomponents
    # ---------------------------------------------------------------------------

    def _generate_opening(self, claim: ClaimExtraction, tone: str) -> str:
        """Constructs an empathetic opening that defuses tension."""
        if claim.threat_level == ThreatLevel.HIGH:
            if tone == "firm_polite":
                return ("Thank you for reaching out to us. I understand that you are seeking an "
                        "urgent resolution regarding your order, and I take your concerns very seriously.")
            elif tone == "empathetic":
                return ("I completely recognize how deeply stressful this situation is, and I sincerely "
                        "apologize for the distress this has caused you. I am here to help guide us to a fair resolution.")
            else:
                return "Thank you for contacting customer support regarding your claim."

        elif claim.sentiment in (Sentiment.ANGRY, Sentiment.FRUSTRATED):
            return ("I can certainly understand how frustrating this experience has been for you, "
                    f"and I appreciate your patience as we review your request regarding your {claim.item}.")

        elif claim.sentiment == Sentiment.PLEADING:
            return ("Thank you for sharing your circumstances with us. We genuinely care about our customers "
                    "and want to ensure your claim is reviewed thoroughly and fairly.")

        else:
            return (f"Thank you for contacting customer service today regarding your "
                    f"{claim.item if claim.item != 'item' else 'recent order'}.")

    def _generate_decision(self, action: ActionType,
                           claim: ClaimExtraction,
                           csp_result: Optional[CSPResult],
                           plan_result: Optional[PlanResult]) -> Tuple[str, List[str]]:
        """Constructs the core policy-grounded decision statement with clause citations."""
        clauses = []
        if csp_result:
            clauses = csp_result.applicable_policies.copy()
        unresolved = self._unresolved_constraints(csp_result)

        if action == ActionType.APPROVE:
            if plan_result and not plan_result.action_approved:
                # Intercepted by Planner Safety Gate
                reason = plan_result.block_reason
                clauses.extend(unresolved)
                return (
                    f"While your {claim.intent} request is in progress, "
                    f"our system has placed a temporary safety hold: {reason} "
                    f"We must complete the remaining verification before proceeding.",
                    clauses
                )
            else:
                return (
                    f"Good news! The required policy checks for your {claim.intent} request "
                    f"have passed. The authorized action is ready to proceed.",
                    clauses
                )

        elif action == ActionType.REJECT:
            violated = csp_result.violated_policies if csp_result else []
            v_str = ", ".join(violated) if violated else "Policy Eligibility Criteria"
            clauses.extend(violated)
            return (
                f"After careful review against our organizational standards, we are unable to approve your "
                f"{claim.intent} request at this time. This determination was made under "
                f"the applicable clause(s) [{v_str}], which indicate that a policy requirement was not met.",
                clauses
            )

        elif action == ActionType.ASK_FOR_EVIDENCE:
            pending = ", ".join(unresolved) or "the required supporting information"
            clauses.extend(unresolved)
            return (
                f"To continue with your {claim.intent} request, we need to complete or verify "
                f"these applicable policy requirements: {pending}.",
                clauses
            )

        elif action == ActionType.INVESTIGATE:
            clauses.extend(unresolved)
            pending = ", ".join(unresolved) or "the applicable policy checks"
            return (
                f"Your {claim.intent} request needs additional verification before we can "
                f"determine the next step. The unresolved policy requirements are: {pending}.",
                clauses
            )

        elif action == ActionType.ESCALATE:
            clauses.extend(unresolved)
            policy_names = ", ".join(csp_result.applicable_policies) if csp_result else ""
            policy_text = f" under {policy_names}" if policy_names else ""
            return (
                f"Your request has been escalated for review{policy_text}. "
                f"The remaining checks are: {', '.join(unresolved) or 'case review'}.",
                clauses
            )

        elif action == ActionType.OFFER_ALTERNATIVE:
            return (
                f"We can review an alternative resolution for your {claim.intent} request "
                "under the applicable policy.",
                clauses
            )

        else:
            return ("Your request is currently being processed by our support specialists.", clauses)

    def _generate_evidence_summary(self, claim: ClaimExtraction,
                                   csp_result: Optional[CSPResult],
                                   risk_assessment: Optional[RiskAssessment]) -> str:
        """Summarizes which evidence was accepted and what remains questioned or missing."""
        graph = csp_result.evidence_graph if csp_result else {}
        provided = graph.get("provided_evidence", [])
        verified = [
            item["type"] for item in provided if item.get("status") == "verified"
        ]
        reported = [
            item["type"] for item in provided if item.get("status") != "verified"
        ]
        parts = []
        if verified:
            parts.append(f"{', '.join(verified)} (verified)")
        if reported:
            parts.append(f"{', '.join(reported)} (reported; validation pending)")
        acc_str = "; ".join(parts) if parts else "None provided or retrieved yet"

        pending = []
        for requirement in graph.get("required_evidence", []):
            status = requirement.get("status", "missing")
            if status in {"missing", "reported"}:
                label = requirement.get("label", requirement.get("field", "Required evidence"))
                suffix = (
                    "not yet provided or retrieved"
                    if status == "missing" else "reported; not validated"
                )
                pending.append(f"{label} ({suffix})")
        if risk_assessment and risk_assessment.risk_factors:
            pending.extend(risk_assessment.risk_factors[:2])
        pending = list(dict.fromkeys(pending))
        pending_text = (
            "; ".join(pending) if pending
            else "No pending evidence requirements identified"
        )

        return (f"Documentation Status:\n"
                f"  • Provided or Retrieved: {acc_str}\n"
                f"  • Pending or Questioned Items: {pending_text}")

    @staticmethod
    def _unresolved_constraints(csp_result: Optional[CSPResult]) -> List[str]:
        if not csp_result:
            return []
        return [
            name for name, status in csp_result.constraints.items()
            if status != ConstraintStatus.SATISFIED
        ]

    def _generate_available_actions(self, action: ActionType,
                                    claim: ClaimExtraction,
                                    csp_result: Optional[CSPResult],
                                    plan_result: Optional[PlanResult]) -> List[str]:
        """Provides concrete, transparent next steps for the customer."""
        actions = []

        if action == ActionType.ASK_FOR_EVIDENCE:
            graph = csp_result.evidence_graph if csp_result else {}
            for requirement in graph.get("required_evidence", []):
                status = requirement.get("status", "missing")
                if status in {"missing", "reported"}:
                    label = requirement.get("label", requirement.get("field", "Required information"))
                    verb = (
                        "Please provide or let us retrieve"
                        if status == "missing" else "Please complete verification for"
                    )
                    actions.append(f"{verb}: {label}.")
                if len(actions) == 3:
                    break
            actions.append("Reply once the requested information is available so we can review the claim again.")

        elif action == ActionType.APPROVE:
            actions.append("Allow 3–5 business days for the funds to reflect in your bank statement.")
            actions.append("Keep your confirmation reference number for your records.")
            if plan_result and not plan_result.action_approved:
                actions.append("Complete the pending return shipment tracking step to release payment.")

        elif action == ActionType.REJECT:
            actions.append("Submit additional documentation if you believe this determination was made in error.")
            actions.append("Request an alternative product exchange under our 14-day Exchange Policy.")
            actions.append("Request a formal escalation to an appeals officer within 7 calendar days.")

        elif action == ActionType.INVESTIGATE:
            actions.append("Allow 24 to 48 hours for our security and logistics teams to complete the audit.")
            actions.append("Ensure your contact phone number and email on file are up to date.")

        elif action == ActionType.ESCALATE:
            actions.append("A supervisory case specialist will review your file and respond within 1 business day.")
            actions.append("You may upload any additional supporting records to this thread in the interim.")

        elif action == ActionType.OFFER_ALTERNATIVE:
            actions.append("Reply 'ACCEPT REPLACEMENT' to initiate an immediate exchange.")
            actions.append("Reply 'STORE CREDIT' to receive an instant digital gift card balance.")

        return actions

    def _generate_closing(self, tone: str) -> str:
        """Standard closing matching the tactical tone."""
        if tone == "empathetic":
            return ("We appreciate your understanding and cooperation as we uphold our commitments "
                    "to service quality and policy fairness. Please let me know how you wish to proceed.")
        else:
            return ("Our team remains committed to assisting you within our established policy framework. "
                    "Please let us know if you have further documentation to submit.")
