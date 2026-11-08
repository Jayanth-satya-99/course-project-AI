# Module 4 — Natural Language Understanding (NLU)
# =================================================
# Rule-based NLU engine that extracts structured information from raw
# customer messages. No external LLM required — uses keyword matching,
# regex patterns, and heuristic scoring.
#
# Extracts:
#   - Intent (refund, exchange, cancellation, warranty, complaint)
#   - Amount ($)
#   - Item name
#   - Purchase date
#   - Order ID
#   - Evidence references
#   - Threat level
#   - Sentiment
#   - Adversarial tactics

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from shared.models import ClaimExtraction, ThreatLevel, Sentiment


class NLUExtractor:
    """
    Rule-based Natural Language Understanding module.

    Parses raw customer text into a structured ClaimExtraction object
    containing claims, dates, amounts, evidence, emotional tone, and
    adversarial tactics.
    """

    # Intent keyword mappings
    INTENT_PATTERNS = {
        "refund": [
            r"\brefund\b", r"\bmoney\s*back\b", r"\breimburse\b",
            r"\breturn\s+(?:the\s+)?(?:item|product|order)\b",
            r"\bget\s+(?:my\s+)?money\b", r"\bcharge\s*back\b",
        ],
        "exchange": [
            r"\bexchange\b", r"\bswap\b", r"\breplace(?:ment)?\b",
            r"\btrade\s+in\b", r"\bswitch\b",
        ],
        "cancellation": [
            r"\bcancel\b", r"\bcancellation\b", r"\bcancel\s+(?:my\s+)?order\b",
            r"\bdon'?t\s+want\s+(?:it|this)\b", r"\bstop\s+(?:the\s+)?order\b",
        ],
        "warranty": [
            r"\bwarranty\b", r"\bguarantee\b", r"\bdefect(?:ive)?\b",
            r"\bmanufactur(?:ing|er)\s+(?:defect|fault)\b",
            r"\bunder\s+warranty\b",
        ],
        "complaint": [
            r"\bcomplain\b", r"\bunhappy\b", r"\bdisappointed\b",
            r"\bterrible\s+(?:service|experience)\b", r"\bawful\b",
        ],
        "return": [
            r"\breturn(?:ing|ed)?\b", r"\bsend\s+(?:it|the item|the product)\s+back\b",
        ],
        "payment": [
            r"\bpayment\b", r"\bpay(?:ment)?\s+for\b", r"\bcharge(?:d)?\b",
        ],
        "delivery": [
            r"\bdelivery\b", r"\bdeliver(?:ed|y)?\b", r"\bproof of delivery\b",
        ],
        "shipment": [
            r"\bshipment\b", r"\bshipping\b", r"\btracking number\b",
        ],
        "order_modification": [
            r"\bchange\s+(?:my\s+)?order\b", r"\bmodify\s+(?:my\s+)?order\b",
            r"\bupdate\s+(?:my\s+)?order\b", r"\bchange the (?:address|quantity|item)\b",
            r"\b(?:change|update|correct)\s+(?:the\s+)?(?:delivery\s+)?(?:address|phone|contact|instructions?)\b",
        ],
        "insurance": [
            r"\binsurance claim\b", r"\binsurance\b", r"\bpolicy payout\b",
        ],
    }

    # Threat detection patterns
    THREAT_PATTERNS = {
        "chargeback_threat": [
            r"\bcharge\s*back\b", r"\bdispute\s+(?:the\s+)?charge\b",
            r"\bcredit\s+card\s+company\b", r"\bbank\s+dispute\b",
        ],
        "legal_threat": [
            r"\bsue\b", r"\blawyer\b", r"\blegal\s+action\b",
            r"\bcourt\b", r"\battorney\b", r"\blawsuit\b",
        ],
        "social_media_threat": [
            r"\bsocial\s+media\b", r"\btwitter\b", r"\breddit\b",
            r"\breview\b", r"\bbad\s+press\b", r"\bgo\s+public\b",
            r"\btell\s+everyone\b",
        ],
        "emotional_appeal": [
            r"\bplease\b.*\bhelp\b", r"\bdesper(?:ate|ation)\b",
            r"\bcan'?t\s+afford\b", r"\bsingle\s+(?:mom|parent|mother|father)\b",
            r"\bkids?\b.*\bbirthday\b", r"\bchristmas\b.*\bgift\b",
            r"\bmedical\b", r"\bemergency\b", r"\bsick\b",
        ],
        "authority_pressure": [
            r"\bmanager\b", r"\bsupervisor\b", r"\bspeak\s+(?:to|with)\s+(?:a\s+)?(?:manager|supervisor)\b",
            r"\bescalate\b", r"\bhigher\s+up\b",
        ],
        "fabricated_evidence": [
            r"\b(?:fake|fabricated|forged|photoshopped|altered)\s+(?:receipt|proof|evidence|photo)\b",
            r"\b(?:receipt|proof)\s+from\s+20\d{2}\b",
        ],
        "contradiction": [],  # Detected via logic, not keywords
        "urgency_pressure": [
            r"\bright\s+now\b", r"\bimmediately\b", r"\basap\b",
            r"\btoday\b", r"\bcan'?t\s+wait\b", r"\burgent\b",
        ],
    }

    # Categorized emotional appeals
    EMOTIONAL_APPEAL_PATTERNS = {
        "medical_hardship": [
            r"\b(?:medical|medicine|hospital|sick|illness|doctor|surgery|emergency room|prescriptions?|health crisis|cancer)\b"
        ],
        "financial_distress": [
            r"\b(?:can'?t afford|cannot afford|broke|no money|starving|paycheck|poverty|rent money|pay my rent|struggling to pay|out of money)\b"
        ],
        "single_parent_family": [
            r"\b(?:single\s+(?:mom|parent|mother|father)|kids?'?\s+birthday|daughter'?s\s+birthday|son'?s\s+birthday|christmas\s+gift|feed\s+my\s+(?:kids|children)|baby\s+food)\b"
        ],
        "bereavement": [
            r"\b(?:funeral|passed\s+away|died|death in(?: the)? family|grieving|grief)\b"
        ],
        "desperation_plea": [
            r"\b(?:please\s+please|desperat(?:e|ion)|begging\s+you|at\s+my\s+wit'?s\s+end|have\s+mercy|crying|ruined\s+my\s+life|beg\s+you)\b"
        ],
    }

    # Categorized threat patterns
    THREAT_CATEGORY_PATTERNS = {
        "legal_action": [
            r"\b(?:sue|suing|lawyer|lawsuit|attorney|court|small\s+claims|legal\s+action|legal\s+counsel)\b"
        ],
        "chargeback": [
            r"\b(?:charge\s*back|dispute\s+(?:the\s+)?charge|credit\s+card\s+company|bank\s+dispute|contact\s+my\s+bank|fraud\s+department)\b"
        ],
        "social_media_exposure": [
            r"\b(?:social\s+media|twitter|reddit|facebook|tiktok|instagram|bad\s+review|ruin\s+your\s+brand|go\s+public|tell\s+everyone|bad\s+press|viral|post\s+all\s+over)\b"
        ],
        "regulatory_complaint": [
            r"\b(?:better\s+business\s+bureau|bbb|ftc|federal\s+trade\s+commission|attorney\s+general|consumer\s+protection|consumer\s+affairs|file\s+a\s+complaint\s+with)\b"
        ],
        "executive_escalation": [
            r"\b(?:speak\s+(?:to|with)\s+(?:a\s+)?(?:manager|supervisor)|speak\s+to\s+(?:your\s+)?manager|talk\s+to\s+a\s+supervisor|escalate|higher\s+up|corporate|ceo|headquarters)\b"
        ],
    }

    # Persuasive conversation strategy patterns
    PERSUASIVE_STRATEGY_PATTERNS = {
        "urgency_pressure": [
            r"\b(?:right\s+now|immediately|asap|today|this\s+second|can'?t\s+wait|urgent|hurry|without\s+delay)\b"
        ],
        "authority_coercion": [
            r"\b(?:manager\s+told\s+me|supervisor\s+(?:dave|said|approved)|already\s+approved\s+by|higher\s+up|corporate|ceo|headquarters)\b"
        ],
        "policy_gaslighting": [
            r"\b(?:supervisor.*approved|already\s+approved|told\s+me\s+to\s+just\s+message|promised\s+(?:me\s+)?a\s+refund|website\s+says\s+(?:100%|unconditional)|consumer\s+law\s+allows|don'?t\s+care\s+(?:about\s+)?your\s+policy|your\s+rules\s+don'?t\s+apply|bend\s+the\s+rule)\b"
        ],
        "reciprocity_claim": [
            r"\b(?:loyal\s+customer|spent\s+thousands|shopped\s+here\s+for\s+years|always\s+buy\s+from\s+you|good\s+customer|regular\s+customer)\b"
        ],
        "feigned_compliance": [
            r"\b(?:already\s+sent|already\s+provided|proof\s+is\s+on\s+file|you\s+already\s+have\s+my\s+receipt|i\s+already\s+showed\s+you)\b"
        ],
    }

    # Evidence keyword dictionary
    EVIDENCE_KEYWORDS = {
        "receipt": ["receipt", "invoice", "proof of purchase", "confirmation email",
                    "order confirmation", "purchase confirmation"],
        "photo": ["photo", "picture", "image", "screenshot", "pic"],
        "video": ["video", "recording"],
        "tracking": ["tracking", "shipping info", "delivery confirmation"],
        "order_id": ["order number", "order id", "order #"],
        "warranty_card": ["warranty card", "warranty certificate"],
        "bank_statement": ["bank statement", "credit card statement",
                           "transaction record"],
        "payment_confirmation": ["payment confirmation", "payment record"],
        "police_report": ["police report", "incident report"],
        "adjuster_report": ["adjuster report", "adjuster assessment",
                            "insurance assessment"],
        "delivery_signature": ["delivery signature", "proof of delivery signature",
                               "signed proof of delivery"],
        "inspection_report": ["inspection report", "inspection result"],
        "return_confirmation": ["return confirmation", "return receipt"],
        "cancellation_confirmation": ["cancellation confirmation",
                                      "cancellation receipt"],
        "refund_confirmation": ["refund confirmation", "refund receipt"],
        "refund_history": ["refund history", "previous refund record"],
        "packaging_tags": ["tags", "tag", "original packaging", "original box", "labels", "unbroken seal"],
        "other_document": ["document", "supporting documents", "official report"],
    }

    # Sentiment keyword mappings
    SENTIMENT_KEYWORDS = {
        Sentiment.THREATENING: [
            "sue", "lawyer", "legal", "court", "chargeback", "dispute",
            "report", "attorney",
        ],
        Sentiment.ANGRY: [
            "ridiculous", "unacceptable", "terrible", "awful", "worst",
            "incompetent", "scam", "ripped off", "furious", "outrageous",
            "disgusting",
        ],
        Sentiment.FRUSTRATED: [
            "frustrated", "annoyed", "disappointed", "unhappy", "upset",
            "waste of time", "again", "still",
        ],
        Sentiment.PLEADING: [
            "please", "beg", "help me", "desperate", "need", "can't afford",
            "really need",
        ],
        Sentiment.COOPERATIVE: [
            "thank", "appreciate", "understand", "of course", "sure",
            "happy to", "no problem", "okay",
        ],
    }

    def extract(self, raw_text: str,
                conversation_history: Optional[List[Dict[str, str]]] = None) -> ClaimExtraction:
        """
        Extract structured claim data from raw customer text and conversation history.

        Args:
            raw_text: The customer's message string.
            conversation_history: Optional list of past dialogue turns [{"role": ..., "content": ...}].

        Returns:
            ClaimExtraction with all extracted fields populated.
        """
        text_lower = raw_text.lower()

        intent = self._extract_intent(text_lower)
        amount = self._extract_amount(raw_text)
        item = self._extract_item(text_lower)
        purchase_date = self._extract_date(raw_text)
        order_id = self._extract_order_id(raw_text)
        claimed_previous_claims = self._extract_previous_claim_count(text_lower)
        evidence = self._extract_evidence(text_lower)

        # Advanced extraction features
        emotional_appeals = self._extract_emotional_appeals(text_lower)
        threats = self._extract_threats(text_lower)
        supporting_evidence = self._extract_supporting_evidence(text_lower, evidence, raw_text)
        contradictory_claims = self._extract_contradictory_claims(
            text_lower, conversation_history, amount, order_id, purchase_date, evidence
        )

        tactics = self._extract_adversarial_tactics(text_lower)

        # Harmonize adversarial tactics with specific threats, appeals, and contradictions
        if "legal_action" in threats and "legal_threat" not in tactics:
            tactics.append("legal_threat")
        if "chargeback" in threats and "chargeback_threat" not in tactics:
            tactics.append("chargeback_threat")
        if "social_media_exposure" in threats and "social_media_threat" not in tactics:
            tactics.append("social_media_threat")
        if ("executive_escalation" in threats or "regulatory_complaint" in threats) and "authority_pressure" not in tactics:
            tactics.append("authority_pressure")
        if emotional_appeals and "emotional_appeal" not in tactics:
            tactics.append("emotional_appeal")
        if contradictory_claims and "contradiction" not in tactics:
            tactics.append("contradiction")

        persuasive_strategies = self._extract_persuasive_strategies(
            text_lower, tactics, emotional_appeals, threats, contradictory_claims
        )
        threat_level = self._assess_threat_level(text_lower, tactics)
        sentiment = self._detect_sentiment(text_lower)

        # Format conversation history
        history_list = [dict(t) for t in (conversation_history or [])]

        return ClaimExtraction(
            intent=intent,
            amount=amount,
            item=item,
            purchase_date=purchase_date,
            order_id=order_id,
            evidence_provided=evidence,
            threat_level=threat_level,
            sentiment=sentiment,
            adversarial_tactics=tactics,
            raw_text=raw_text,
            claimed_previous_claims=claimed_previous_claims,
            contradictory_claims=contradictory_claims,
            supporting_evidence=supporting_evidence,
            emotional_appeals=emotional_appeals,
            threats=threats,
            persuasive_strategies=persuasive_strategies,
            conversation_history=history_list,
        )

    def _extract_intent(self, text: str) -> str:
        """Detect the primary intent from the message."""
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
            if score > 0:
                scores[intent] = score

        if scores:
            return max(scores, key=scores.get)
        return "complaint"  # default

    def _extract_amount(self, text: str) -> float:
        """Extract monetary amounts from the text."""
        # Match patterns like $500, $49.99, 500 dollars, etc.
        patterns = [
            r'\$\s*([\d,]+(?:\.\d{2})?)',
            r'([\d,]+(?:\.\d{2})?)\s*(?:dollars|usd|bucks)',
            r'(?:cost|paid|worth|price|amount)\s*(?:of\s*)?\$?\s*([\d,]+(?:\.\d{2})?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(",", "")
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        return 0.0

    def _extract_item(self, text: str) -> str:
        """Extract the product/item name from the text."""
        products = [
            "headphones", "television", "laptop", "camera", "tablet", "speaker",
            "monitor", "keyboard", "printer", "router", "blender", "toaster",
            "appliance", "jacket", "shirt", "dress", "shoes", "phone", "watch",
            "mouse", "console", "game", "bag", "tv",
        ]
        for product in products:
            if re.search(rf"\b{re.escape(product)}\b", text):
                return product

        # Common product patterns
        item_patterns = [
            r'(?:my|the|a)\s+([\w\s]+?)(?:\s+(?:is|was|broke|doesn\'t|stopped|arrived))',
            r'(?:bought|purchased|ordered)\s+(?:a|an|the)?\s*([\w\s]+?)(?:\s+(?:from|on|for|and))',
            r'(?:return|refund|exchange)\s+(?:the|my|a)?\s*([\w\s]+?)(?:\s|$|\.)',
        ]
        for pattern in item_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                item = match.group(1).strip()
                if len(item) > 2 and len(item) < 50:
                    return item

        return "item"

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract purchase date from the text."""
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',                    # 2026-10-01
            r'(\d{1,2}/\d{1,2}/\d{4})',                # 10/01/2026
            r'(\d{1,2}/\d{1,2}/\d{2})',                # 10/01/26
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})',
            r'(\d{1,2}\s+(?:hours?|days?|weeks?|months?)\s+ago)',
            r'\b(yesterday|today|last\s+week|last\s+month)\b',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Convert relative dates
                if "ago" in date_str.lower():
                    return self._parse_relative_date(date_str)
                relative_days = {
                    "today": 0,
                    "yesterday": 1,
                    "last week": 7,
                    "last month": 30,
                }
                if date_str.lower() in relative_days:
                    return (datetime.now() - timedelta(
                        days=relative_days[date_str.lower()]
                    )).strftime("%Y-%m-%d")
                return date_str
        return None

    def _parse_relative_date(self, text: str) -> str:
        """Convert relative date expressions to absolute dates."""
        match = re.search(r'(\d+)\s+(hours?|days?|weeks?|months?)', text, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            unit = match.group(2).lower()
            if "hour" in unit:
                date = datetime.now() - timedelta(hours=num)
                return date.strftime("%Y-%m-%d %H:%M:%S")
            if "week" in unit:
                num *= 7
            elif "month" in unit:
                num *= 30
            date = datetime.now() - timedelta(days=num)
            return date.strftime("%Y-%m-%d")
        return None

    def _extract_order_id(self, text: str) -> Optional[str]:
        """Extract order ID from the text."""
        patterns = [
            r'\b(?:order\s*(?:#|number|id)\s*[:#]?\s*)(ORD-\d+)\b',
            r'\b(ORD-\d+)\b',
            r'#(\w{6,})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def _extract_previous_claim_count(self, text: str) -> Optional[int]:
        """Extract a customer-reported claim count; this remains unverified."""
        patterns = (
            r"\b(?:returned|refunded|filed|made)\s+(\d+)\s+(?:items?|claims?|returns?)\b",
            r"\b(\d+)\s+(?:items?|claims?|returns?)\s+(?:already|this\s+month|in\s+the\s+past)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_evidence(self, text: str) -> List[str]:
        """Detect what evidence the customer claims to have."""
        evidence = []
        for evidence_type, keywords in self.EVIDENCE_KEYWORDS.items():
            found = any(kw in text for kw in keywords)
            negated = any(
                re.search(
                    rf"\b(?:no|without|not|don't|do not|can't|cannot)\s+"
                    rf"(?:\w+\s+){{0,4}}(?:{re.escape(keyword)})\b",
                    text,
                    re.IGNORECASE,
                )
                for keyword in keywords
            )
            if found and not negated:
                evidence.append(evidence_type)
        return evidence

    def _extract_supporting_evidence(
        self, text_lower: str, evidence_types: List[str], raw_text: str
    ) -> List[Dict[str, Any]]:
        """Construct structured evidence objects with matched snippets."""
        supporting = []
        for ev in evidence_types:
            snippet = ""
            keywords = self.EVIDENCE_KEYWORDS.get(ev, [ev])
            for kw in keywords:
                pos = text_lower.find(kw.lower())
                if pos != -1:
                    start = max(0, pos - 30)
                    end = min(len(raw_text), pos + len(kw) + 30)
                    snippet = raw_text[start:end].strip()
                    break
            supporting.append({
                "evidence_type": ev,
                "description": f"Customer referenced {ev.replace('_', ' ')}",
                "snippet": snippet or ev,
                "confidence": 0.95,
                "status": "reported",
            })
        return supporting

    def _extract_emotional_appeals(self, text: str) -> List[str]:
        """Detect specific categories of emotional appeals."""
        appeals = []
        for appeal_type, patterns in self.EMOTIONAL_APPEAL_PATTERNS.items():
            if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                appeals.append(appeal_type)
        return appeals

    def _extract_threats(self, text: str) -> List[str]:
        """Detect specific categories of adversarial threats."""
        threats = []
        for threat_type, patterns in self.THREAT_CATEGORY_PATTERNS.items():
            if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                threats.append(threat_type)
        return threats

    def _extract_contradictory_claims(
        self,
        text_lower: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        current_amount: float = 0.0,
        current_order_id: Optional[str] = None,
        current_date: Optional[str] = None,
        current_evidence: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect contradictions within the message and across conversation turns."""
        contradictions = []

        # 1. Intra-message statement conflicts
        unopened_match = re.search(
            r"\b(?:unopened|never\s+opened|sealed(?:\s+box)?|factory\s+sealed|intact\s+box|brand\s+new\s+in\s+box)\b",
            text_lower,
        )
        damaged_match = re.search(
            r"\b(?:cracked|shattered|broken\s+screen|scratched|dented|defective\s+screen|physically\s+damaged|damaged\s+inside)\b",
            text_lower,
        )
        if unopened_match and damaged_match:
            contradictions.append({
                "type": "claim_statement_conflict",
                "source": "internal",
                "claim_a": unopened_match.group(0),
                "claim_b": damaged_match.group(0),
                "description": (
                    "Customer claims the package was sealed and unopened while also "
                    "alleging interior or screen damage."
                ),
            })

        not_received_match = re.search(
            r"\b(?:never\s+received|never\s+arrived|did\s+not\s+receive|not\s+delivered|package\s+lost|lost\s+in\s+transit)\b",
            text_lower,
        )
        returning_match = re.search(
            r"\b(?:return\s+(?:the|this)\s+(?:item|product)|send\s+(?:it|the\s+item)\s+back|returning\s+(?:the|this)\s+(?:item|product)|have\s+the\s+item)\b",
            text_lower,
        )
        if not_received_match and returning_match:
            contradictions.append({
                "type": "claim_statement_conflict",
                "source": "internal",
                "claim_a": not_received_match.group(0),
                "claim_b": returning_match.group(0),
                "description": (
                    "Customer claims the package never arrived while simultaneously "
                    "requesting to return the physical item."
                ),
            })

        no_receipt_match = re.search(
            r"\b(?:no\s+receipt|without\s+a\s+receipt|lost\s+(?:the\s+|my\s+)?receipt|don'?t\s+have\s+(?:a\s+)?receipt)\b",
            text_lower,
        )
        have_receipt_match = re.search(
            r"\b(?:have\s+(?:the|my|a)\s+receipt|attached\s+receipt|receipt\s+is\s+attached)\b",
            text_lower,
        )
        if no_receipt_match and have_receipt_match:
            contradictions.append({
                "type": "evidence_conflict",
                "source": "internal",
                "claim_a": have_receipt_match.group(0),
                "claim_b": no_receipt_match.group(0),
                "description": (
                    "Customer claims to possess the receipt while stating in the same message that they have no receipt."
                ),
            })

        # 2. Cross-turn conflicts across conversation history
        if conversation_history:
            for turn in conversation_history:
                if turn.get("role") != "user":
                    continue
                prev_text = turn.get("content", "")
                prev_lower = prev_text.lower()

                # Amount mismatch across turns
                prev_amt = self._extract_amount(prev_text)
                if prev_amt > 0 and current_amount > 0 and abs(prev_amt - current_amount) > 0.01:
                    contradictions.append({
                        "type": "amount_mismatch",
                        "source": "cross_turn",
                        "claim_a": f"${prev_amt:.2f}",
                        "claim_b": f"${current_amount:.2f}",
                        "description": f"Claimed amount changed from ${prev_amt:.2f} to ${current_amount:.2f} across conversation turns.",
                    })

                # Order mismatch across turns
                prev_order = self._extract_order_id(prev_text)
                if prev_order and current_order_id and prev_order != current_order_id:
                    contradictions.append({
                        "type": "order_mismatch",
                        "source": "cross_turn",
                        "claim_a": prev_order,
                        "claim_b": current_order_id,
                        "description": f"Order identifier shifted from {prev_order} to {current_order_id} across conversation turns.",
                    })

                # Evidence contradiction across turns
                prev_evidence = self._extract_evidence(prev_lower)
                if "receipt" in prev_evidence and no_receipt_match:
                    contradictions.append({
                        "type": "evidence_conflict",
                        "source": "cross_turn",
                        "claim_a": "Previously claimed to have receipt",
                        "claim_b": "Currently states no receipt available",
                        "description": "Customer previously claimed to have a receipt, but now states they do not have one.",
                    })

        return contradictions

    def _extract_persuasive_strategies(
        self,
        text_lower: str,
        tactics: List[str],
        emotional_appeals: List[str],
        threats: List[str],
        contradictory_claims: List[Dict[str, Any]],
    ) -> List[str]:
        """Identify conversational persuasion and manipulation strategies."""
        strategies = set()

        for strategy, patterns in self.PERSUASIVE_STRATEGY_PATTERNS.items():
            if any(re.search(p, text_lower, re.IGNORECASE) for p in patterns):
                strategies.add(strategy)

        if "urgency_pressure" in tactics:
            strategies.add("urgency_pressure")
        if emotional_appeals or "emotional_appeal" in tactics:
            strategies.add("sympathy_exploitation")
        if "legal_action" in threats or "chargeback" in threats or "legal_threat" in tactics or "chargeback_threat" in tactics:
            strategies.add("threat_intimidation")
        if "social_media_exposure" in threats or "social_media_threat" in tactics:
            strategies.add("reputation_damage")
        if "executive_escalation" in threats or "regulatory_complaint" in threats or "authority_pressure" in tactics:
            strategies.add("authority_coercion")
        if contradictory_claims or "contradiction" in tactics:
            strategies.add("fact_shifting")

        return sorted(strategies)

    def _extract_adversarial_tactics(self, text: str) -> List[str]:
        """Detect adversarial/manipulative tactics in the message."""
        detected = []
        for tactic, patterns in self.THREAT_PATTERNS.items():
            if patterns:  # Skip empty pattern lists
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        detected.append(tactic)
                        break
        return detected

    def _assess_threat_level(self, text: str,
                             tactics: List[str]) -> ThreatLevel:
        """Compute overall threat level from detected tactics."""
        high_threat_tactics = {"legal_threat", "chargeback_threat"}
        medium_threat_tactics = {"social_media_threat", "authority_pressure",
                                 "urgency_pressure"}
        low_threat_tactics = {"emotional_appeal"}

        has_high = bool(high_threat_tactics & set(tactics))
        has_medium = bool(medium_threat_tactics & set(tactics))
        has_low = bool(low_threat_tactics & set(tactics))

        # Exclamation marks and caps also increase threat
        exclamation_count = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        if has_high or (has_medium and exclamation_count > 2):
            return ThreatLevel.HIGH
        elif has_medium or (has_low and caps_ratio > 0.3):
            return ThreatLevel.MEDIUM
        elif has_low or exclamation_count > 1:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _detect_sentiment(self, text: str) -> Sentiment:
        """Detect the dominant emotional sentiment."""
        scores = {}
        for sentiment, keywords in self.SENTIMENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[sentiment] = score

        if scores:
            return max(scores, key=scores.get)
        return Sentiment.NEUTRAL
