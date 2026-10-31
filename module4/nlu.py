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
from typing import List, Optional, Tuple

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

    def extract(self, raw_text: str) -> ClaimExtraction:
        """
        Extract structured claim data from raw customer text.

        Args:
            raw_text: The customer's message string.

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
        tactics = self._extract_adversarial_tactics(text_lower)
        threat_level = self._assess_threat_level(text_lower, tactics)
        sentiment = self._detect_sentiment(text_lower)

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
        evidence_keywords = {
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
            "other_document": ["document", "supporting documents", "official report"],
        }
        for evidence_type, keywords in evidence_keywords.items():
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
