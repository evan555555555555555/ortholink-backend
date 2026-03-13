"""
OrthoLink -- Certificate of Free Sale (CFS) Verifier

Semantic verifier for Certificate of Free Sale documents.

CRITICAL REGULATORY CONTEXT:
  A CFS (also called Certificate of Free Commerce / Export Certificate) is issued
  by a National Competent Authority (NCA) or government body confirming that a
  medical device is legally sold/marketed in the country of manufacture/origin.
  It is a REGULATORY DOCUMENT — not a commercial document.

  It must NEVER be confused with:
    - Commercial invoices (billing documents)
    - Letters of Authorization / distributor agreements (commercial contracts)
    - CE certificates (conformity assessment certificates)
    - ISO 13485 certificates (quality management system certificates)
    - Distribution agreements (private commercial contracts)

  Misclassification creates serious regulatory risk when importing into:
    KSA (SFDA), India (CDSCO), Brazil (ANVISA), China (NMPA), etc.,
  all of which require a valid CFS before granting device registration.

Usage:
    from app.services.cfs_verifier import CFSVerifier
    verifier = CFSVerifier()
    result = await verifier.verify_cfs(document_text, claimed_country="US")
    reqs = verifier.check_cfs_requirements("IN")
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — CFS keyword signals
# ---------------------------------------------------------------------------

# Positive CFS signals — strong indicators of a genuine CFS
_CFS_POSITIVE_SIGNALS: list[str] = [
    "certificate of free sale",
    "certificate of free commerce",
    "certificate of exportation",
    "certificate of free distribution",
    "freely sold",
    "free sale",
    "free distribution",
    "free commerce",
    "freely marketed",
    "legally marketed",
    "authorized for sale",
    "authorised for sale",
    "no objection",
    "competent authority",
    "national authority",
    "ministry of health",
    "department of health",
    "government of",
    "hereby certifies",
    "this is to certify",
    "is freely available",
    "conformity to standards",
    "complies with applicable regulations",
    "registered medical device",
]

# Negative CFS signals — strong indicators of a NON-CFS document
_NON_CFS_SIGNALS: list[str] = [
    "purchase order",
    "invoice",
    "bill of lading",
    "commercial invoice",
    "pro forma invoice",
    "letter of credit",
    "distribution agreement",
    "distribution contract",
    "agency agreement",
    "authorization letter",
    "letter of authorization",
    "power of attorney",
    "ce certificate",
    "iso 13485",
    "quality management",
    "notified body",
    "conformity assessment",
    "terms and conditions",
    "payment terms",
    "unit price",
    "total amount",
    "bank account",
    "wire transfer",
]

# Document type classification
_DOC_TYPE_MAP: dict[str, str] = {
    "certificate_of_free_sale": "CFS",
    "commercial_invoice": "COMMERCIAL_INVOICE",
    "letter_of_authorization": "LETTER_OF_AUTHORIZATION",
    "distribution_agreement": "DISTRIBUTION_AGREEMENT",
    "ce_certificate": "CE_CERTIFICATE",
    "iso_certificate": "ISO_CERTIFICATE",
    "other": "OTHER",
}

# Required CFS fields per target country
# These are the fields an importing country's NCA expects to see in a CFS
_CFS_REQUIRED_FIELDS: dict[str, list[str]] = {
    "IN": [  # CDSCO India
        "issuing_authority",
        "device_name",
        "manufacturer_name",
        "manufacturer_address",
        "country_of_origin",
        "registration_number",
        "freely_sold_statement",
        "signature_and_seal",
        "date_of_issue",
    ],
    "SA": [  # SFDA Saudi Arabia
        "issuing_authority",
        "device_name",
        "manufacturer_name",
        "manufacturer_address",
        "country_of_origin",
        "device_class",
        "freely_sold_statement",
        "apostille_or_notarisation",
        "signature_and_seal",
        "date_of_issue",
        "validity_period",
    ],
    "BR": [  # ANVISA Brazil
        "issuing_authority",
        "device_name",
        "manufacturer_name",
        "country_of_manufacture",
        "registration_number",
        "freely_sold_statement",
        "consularisation",
        "signature_and_seal",
        "date_of_issue",
    ],
    "CN": [  # NMPA China
        "issuing_authority",
        "device_name",
        "manufacturer_name",
        "manufacturer_address",
        "country_of_origin",
        "freely_sold_statement",
        "notarisation",
        "signature_and_seal",
        "date_of_issue",
    ],
    "MX": [  # COFEPRIS Mexico
        "issuing_authority",
        "device_name",
        "manufacturer_name",
        "country_of_origin",
        "freely_sold_statement",
        "signature_and_seal",
        "date_of_issue",
    ],
    "UA": [  # Ukraine MOH
        "issuing_authority",
        "device_name",
        "manufacturer_name",
        "country_of_origin",
        "freely_sold_statement",
        "signature_and_seal",
        "date_of_issue",
    ],
    # Default for unlisted countries
    "_default": [
        "issuing_authority",
        "device_name",
        "manufacturer_name",
        "country_of_origin",
        "freely_sold_statement",
        "date_of_issue",
    ],
}

# Recognised issuing authorities per country (partial list for key NCA names)
_RECOGNISED_AUTHORITIES: dict[str, list[str]] = {
    "US": [
        "food and drug administration",
        "fda",
        "u.s. department of health and human services",
        "department of commerce",
        "international trade administration",
    ],
    "EU": [
        "european commission",
        "competent authority",
        "ministry of health",
    ],
    "UK": [
        "medicines and healthcare products regulatory agency",
        "mhra",
        "department of health and social care",
    ],
    "AU": [
        "therapeutic goods administration",
        "tga",
        "department of health",
    ],
    "CA": [
        "health canada",
        "santé canada",
        "government of canada",
    ],
    "DE": [
        "bundesinstitut für arzneimittel und medizinprodukte",
        "bfarm",
        "zentralstelle der länder",
    ],
    "FR": [
        "agence nationale de sécurité du médicament",
        "ansm",
        "ministère de la santé",
    ],
    "JP": [
        "pharmaceuticals and medical devices agency",
        "pmda",
        "ministry of health, labour and welfare",
        "mhlw",
    ],
    "IN": [
        "central drugs standard control organisation",
        "cdsco",
        "ministry of health and family welfare",
    ],
    "_default": [
        "ministry of health",
        "department of health",
        "health authority",
        "regulatory authority",
        "competent authority",
        "government",
    ],
}

# Field detection patterns
_FIELD_PATTERNS: dict[str, list[str]] = {
    "issuing_authority": [
        r"issued\s+by",
        r"issuing\s+authority",
        r"ministry\s+of\s+health",
        r"department\s+of\s+health",
        r"competent\s+authority",
        r"government\s+of",
    ],
    "device_name": [
        r"device\s+name",
        r"product\s+name",
        r"medical\s+device",
        r"device\s+description",
        r"product\s+description",
    ],
    "manufacturer_name": [
        r"manufacturer",
        r"manufactured\s+by",
        r"company\s+name",
        r"applicant",
    ],
    "manufacturer_address": [
        r"manufacturer.*address",
        r"address.*manufacturer",
        r"place\s+of\s+manufacture",
        r"\d+.*street|avenue|road|blvd",
    ],
    "country_of_origin": [
        r"country\s+of\s+(origin|manufacture)",
        r"manufactured\s+in",
        r"origin\s+country",
    ],
    "registration_number": [
        r"registration\s+number",
        r"license\s+number",
        r"authorisation\s+number",
        r"510\(k\)|pma|artg|ukca",
        r"device\s+license",
    ],
    "freely_sold_statement": [
        r"freely\s+sold",
        r"free\s+sale",
        r"free\s+distribution",
        r"legally\s+marketed",
        r"no\s+objection",
        r"authorized\s+for\s+sale",
        r"freely\s+available",
    ],
    "date_of_issue": [
        r"date\s+of\s+issue",
        r"issued\s+on",
        r"issue\s+date",
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{4}-\d{2}-\d{2}",
    ],
    "signature_and_seal": [
        r"signed\s+by",
        r"authorized\s+signatory",
        r"authorised\s+signatory",
        r"stamp",
        r"seal",
        r"official\s+seal",
        r"signature",
    ],
    "validity_period": [
        r"valid\s+(until|through|for)",
        r"expiry\s+date",
        r"expiration",
        r"validity",
    ],
    "device_class": [
        r"device\s+class",
        r"classification",
        r"class\s+(i|ii|iii|iv)\b",
    ],
    "apostille_or_notarisation": [
        r"apostille",
        r"notari[sz]",
        r"legali[sz]",
        r"authenticated",
    ],
    "consularisation": [
        r"consular",
        r"consulariz",
        r"legali[sz]ed\s+by",
    ],
    "notarisation": [
        r"notari[sz]",
        r"notary\s+public",
    ],
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DocumentClassification(BaseModel):
    """Result of classifying a document's type."""

    doc_type: str = Field(..., description="Canonical document type code (e.g. CFS, COMMERCIAL_INVOICE)")
    confidence: float = Field(..., description="Classification confidence 0.0–1.0")
    signals: list[str] = Field(
        default_factory=list, description="Text signals that drove the classification"
    )
    positive_signal_count: int = Field(default=0)
    negative_signal_count: int = Field(default=0)


class CFSRequirements(BaseModel):
    """Required and optional CFS fields for a target country."""

    country: str
    required_fields: list[str]
    optional_fields: list[str] = Field(default_factory=list)
    recognised_authorities: list[str] = Field(default_factory=list)
    notes: str = Field(default="")


class CFSVerification(BaseModel):
    """Complete verification result for a CFS document."""

    is_valid: bool = Field(..., description="Whether this is a valid CFS for the claimed country")
    confidence: float = Field(..., description="Overall confidence score 0.0–1.0")
    document_type: str = Field(..., description="Classified document type")
    issuing_authority: str = Field(default="", description="Detected issuing authority")
    missing_fields: list[str] = Field(
        default_factory=list, description="Required CFS fields not detected in document"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Regulatory warnings for this document"
    )
    country: str = Field(..., description="Target country code for CFS requirements")
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    classification: Optional[DocumentClassification] = None
    faiss_context: list[str] = Field(
        default_factory=list, description="Relevant regulatory context from FAISS"
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CFSVerifier:
    """
    Semantic verifier for Certificate of Free Sale documents.

    CRITICAL: A CFS is a regulatory document — not a commercial document.
    Misclassification creates serious regulatory import risk.

    Verification pipeline:
      1. Classify document type via keyword + semantic signal scoring
      2. Detect issuing authority
      3. Validate required CFS fields per target country
      4. Check issuing authority against recognised NCA list
      5. Compute confidence score
      6. Query FAISS for regulatory context (country-specific CFS requirements)
    """

    def __init__(self) -> None:
        self._vector_store: Any = None

    def _get_vector_store(self) -> Any:
        if self._vector_store is None:
            from app.tools.vector_store import get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store

    # -- Text normalisation -------------------------------------------------

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Normalise whitespace and lowercase for matching."""
        return re.sub(r"\s+", " ", text.lower().strip())

    # -- Document classification --------------------------------------------

    def classify_document_type(self, text: str) -> DocumentClassification:
        """
        Classify the document type via keyword signal scoring.

        Scoring:
          - Each positive CFS signal matched: +2.0 points
          - Each negative CFS signal matched: -3.0 (stronger penalty for non-CFS)
        Final score > 2 = CFS; otherwise classify as most likely non-CFS type.
        """
        norm = self._normalise_text(text)
        score = 0.0
        matched_positive: list[str] = []
        matched_negative: list[str] = []

        for signal in _CFS_POSITIVE_SIGNALS:
            if signal in norm:
                score += 2.0
                matched_positive.append(signal)

        for signal in _NON_CFS_SIGNALS:
            if signal in norm:
                score -= 3.0
                matched_negative.append(signal)

        # Determine doc_type
        if score > 2.0:
            doc_type = "CFS"
            confidence = min(0.5 + (score / 20.0), 0.99)
        elif score <= -6.0:
            # Strong non-CFS signal: classify by most-matched category
            if any(s in norm for s in ("invoice", "purchase order", "bill of lading")):
                doc_type = "COMMERCIAL_INVOICE"
            elif any(s in norm for s in ("distribution agreement", "distribution contract", "agency agreement")):
                doc_type = "DISTRIBUTION_AGREEMENT"
            elif any(s in norm for s in ("authorization letter", "letter of authorization", "power of attorney")):
                doc_type = "LETTER_OF_AUTHORIZATION"
            elif any(s in norm for s in ("ce certificate", "notified body")):
                doc_type = "CE_CERTIFICATE"
            elif any(s in norm for s in ("iso 13485", "quality management")):
                doc_type = "ISO_CERTIFICATE"
            else:
                doc_type = "OTHER"
            confidence = min(0.4 + abs(score) / 20.0, 0.95)
        else:
            doc_type = "AMBIGUOUS"
            confidence = 0.3

        return DocumentClassification(
            doc_type=doc_type,
            confidence=round(confidence, 3),
            signals=matched_positive + [f"[NON-CFS] {s}" for s in matched_negative],
            positive_signal_count=len(matched_positive),
            negative_signal_count=len(matched_negative),
        )

    # -- Field detection ----------------------------------------------------

    def _detect_present_fields(self, text: str) -> dict[str, bool]:
        """
        Detect which structured CFS fields are present in the document text.

        Uses regex patterns from _FIELD_PATTERNS. Returns dict of field_name -> found.
        """
        norm = self._normalise_text(text)
        detected: dict[str, bool] = {}
        for field, patterns in _FIELD_PATTERNS.items():
            detected[field] = any(
                bool(re.search(pat, norm, re.IGNORECASE)) for pat in patterns
            )
        return detected

    def validate_cfs_fields(
        self, parsed_fields: dict[str, bool], country: str
    ) -> list[str]:
        """
        Validate that all required CFS fields for the target country are present.

        Args:
            parsed_fields: Dict of field_name -> bool (True = detected)
            country:       ISO 3166-1 alpha-2 country code

        Returns:
            List of missing required field names.
        """
        reqs = _CFS_REQUIRED_FIELDS.get(country.upper(), _CFS_REQUIRED_FIELDS["_default"])
        return [field for field in reqs if not parsed_fields.get(field, False)]

    # -- Issuing authority detection ----------------------------------------

    def _detect_issuing_authority(self, text: str, country: str) -> str:
        """
        Extract the issuing authority name from document text.

        Returns the matched authority name or empty string if not found.
        """
        norm = self._normalise_text(text)
        authorities = self.get_recognised_authorities(country)
        for authority in authorities:
            if authority.lower() in norm:
                return authority
        # Fallback: look for "issued by:" or "issuing authority:" followed by text
        match = re.search(
            r"(?:issued\s+by|issuing\s+authority|authority)[:.]?\s*([A-Z][^.\n]{5,60})",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""

    # -- Recognised authorities per country --------------------------------

    def get_recognised_authorities(self, country: str) -> list[str]:
        """
        Return a list of recognised national competent authorities for a country.

        Combines country-specific and default lists.
        """
        country_upper = country.upper()
        specific = _RECOGNISED_AUTHORITIES.get(country_upper, [])
        defaults = _RECOGNISED_AUTHORITIES["_default"]
        return list(dict.fromkeys(specific + defaults))  # dedup, preserve order

    # -- CFS requirements lookup --------------------------------------------

    def check_cfs_requirements(self, country: str) -> CFSRequirements:
        """
        Return the CFS requirements for a target import country.

        Args:
            country: ISO 3166-1 alpha-2 country code

        Returns:
            CFSRequirements with required_fields, optional_fields, authorities, notes.
        """
        country_upper = country.upper()
        required = _CFS_REQUIRED_FIELDS.get(country_upper, _CFS_REQUIRED_FIELDS["_default"])
        all_possible = list(_FIELD_PATTERNS.keys())
        optional = [f for f in all_possible if f not in required]
        authorities = self.get_recognised_authorities(country_upper)

        country_notes: dict[str, str] = {
            "SA": (
                "SFDA requires CFS to be apostilled/notarised AND legalised by the Saudi embassy. "
                "Validity typically 1 year from issue date."
            ),
            "BR": (
                "ANVISA requires CFS to be consularised at the Brazilian consulate in the country of manufacture. "
                "Document must be in Portuguese or accompanied by certified translation."
            ),
            "CN": (
                "NMPA requires CFS to be notarised and authenticated. "
                "Chinese translation by certified translator required."
            ),
            "IN": (
                "CDSCO requires CFS from the competent authority of the country of origin/manufacture. "
                "Must be on official letterhead with original signature and seal."
            ),
            "UA": (
                "Ukraine MOH requires CFS as part of the device registration dossier. "
                "Ukrainian translation may be required."
            ),
        }

        return CFSRequirements(
            country=country_upper,
            required_fields=required,
            optional_fields=optional,
            recognised_authorities=authorities,
            notes=country_notes.get(country_upper, ""),
        )

    # -- FAISS context query ------------------------------------------------

    async def _get_faiss_context(
        self, country: str, top_k: int = 5
    ) -> list[str]:
        """
        Query FAISS for regulatory context about CFS requirements in the given country.

        Uses the country partition of the vector store.
        """
        try:
            store = self._get_vector_store()
            query = f"certificate of free sale requirements import registration {country}"
            results = store.search(query=query, country=country, top_k=top_k)
            return [
                f"{r.get('regulation_name', '')} {r.get('article', '')} — {r.get('text', '')[:150]}"
                for r in results
                if r.get("text")
            ]
        except Exception as exc:
            logger.debug("cfs_verifier FAISS query failed: %s", exc)
            return []

    # -- Main verification --------------------------------------------------

    async def verify_cfs(
        self, document_text: str, claimed_country: str
    ) -> CFSVerification:
        """
        Verify a document is a valid Certificate of Free Sale for the claimed country.

        Verification pipeline:
          1. Classify document type
          2. Detect present fields
          3. Validate required fields for country
          4. Detect issuing authority
          5. Check authority is recognised
          6. Compute overall confidence
          7. Query FAISS for regulatory context

        Args:
            document_text:  Full text of the document
            claimed_country: ISO 3166-1 alpha-2 code of the claimed issuing country

        Returns:
            CFSVerification with is_valid, confidence, missing_fields, and warnings.
        """
        country_upper = claimed_country.upper()
        now = datetime.now(timezone.utc)
        warnings: list[str] = []

        # Step 1: classify document type
        classification = self.classify_document_type(document_text)

        # Step 2: detect present fields
        detected_fields = self._detect_present_fields(document_text)

        # Step 3: validate required fields
        missing_fields = self.validate_cfs_fields(detected_fields, country_upper)

        # Step 4: detect issuing authority
        issuing_authority = self._detect_issuing_authority(document_text, country_upper)

        # Step 5: check authority is recognised
        recognised = self.get_recognised_authorities(country_upper)
        authority_recognised = bool(issuing_authority) and any(
            r.lower() in issuing_authority.lower() for r in recognised
        )

        # Step 6: compute overall confidence and validity
        is_cfs = classification.doc_type == "CFS"
        base_confidence = classification.confidence if is_cfs else 0.1

        # Penalty for missing fields
        all_required = _CFS_REQUIRED_FIELDS.get(country_upper, _CFS_REQUIRED_FIELDS["_default"])
        field_completion = 1.0 - (len(missing_fields) / max(len(all_required), 1))
        adjusted_confidence = round(base_confidence * field_completion, 3)

        # Authority bonus
        if authority_recognised:
            adjusted_confidence = min(adjusted_confidence + 0.05, 0.99)

        is_valid = is_cfs and len(missing_fields) == 0 and adjusted_confidence >= 0.55

        # Build warnings
        if not is_cfs:
            warnings.append(
                f"MISCLASSIFICATION RISK: Document appears to be a {classification.doc_type}, "
                "not a Certificate of Free Sale. Do not submit to regulatory authority."
            )
        if missing_fields:
            warnings.append(
                f"Missing required fields for {country_upper}: {', '.join(missing_fields)}. "
                "CFS may be rejected by the importing country NCA."
            )
        if not authority_recognised and issuing_authority:
            warnings.append(
                f"Issuing authority '{issuing_authority}' not found in recognised NCA list for {country_upper}. "
                "Verify with target country regulatory authority."
            )
        if not issuing_authority:
            warnings.append(
                "Issuing authority not detected. CFS must be issued on official government letterhead."
            )
        if adjusted_confidence < 0.55:
            warnings.append(
                "Low overall confidence. Manual review by qualified regulatory affairs professional required."
            )

        # Step 7: FAISS context
        faiss_context = await self._get_faiss_context(country_upper)

        return CFSVerification(
            is_valid=is_valid,
            confidence=adjusted_confidence,
            document_type=classification.doc_type,
            issuing_authority=issuing_authority,
            missing_fields=missing_fields,
            warnings=warnings,
            country=country_upper,
            verified_at=now,
            classification=classification,
            faiss_context=faiss_context,
        )
