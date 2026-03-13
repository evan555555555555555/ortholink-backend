"""
OrthoLink -- Legacy Device Certificate Mapper

Maps MDD/AIMDD/IVDD certificates to EU MDR 2017/745 requirements.

Regulatory context:
  - EU MDR 2017/745 replaced MDD 93/42/EEC and AIMDD 90/385/EEC
  - EU IVDR 2017/746 replaced IVDD 98/79/EC
  - Transition deadlines per EU Regulation 2024/1860 (amended):
      * Class III / implantable: sell-off by May 26 2026
      * Class IIb non-implantable: until Dec 31 2027
      * Class IIa + Class I sterile/measuring: until Dec 31 2028
  - Legacy certificates remain valid until (expiry OR transition deadline), whichever is earlier

Usage:
    from app.services.legacy_mapper import LegacyMapper
    mapper = LegacyMapper()
    mapping = await mapper.map_mdd_to_mdr("EC_2019_12345_ABC", "BSI")
    audit = await mapper.audit_legacy_portfolio(["EC_2019_12345_ABC", "EC_2020_99999_XYZ"])
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — Transition deadlines per EU Regulation 2024/1860
# ---------------------------------------------------------------------------

# (cert_type, device_class) → final sell-off deadline
_TRANSITION_DEADLINES: dict[tuple[str, str], date] = {
    # MDD
    ("MDD", "III"): date(2026, 5, 26),
    ("MDD", "IIb"): date(2027, 12, 31),
    ("MDD", "IIa"): date(2028, 12, 31),
    ("MDD", "I_sterile"): date(2028, 12, 31),
    ("MDD", "I_measuring"): date(2028, 12, 31),
    ("MDD", "I"): date(2028, 12, 31),
    # AIMDD (Active Implantable Medical Devices — always Class III equivalent)
    ("AIMDD", "III"): date(2026, 5, 26),
    ("AIMDD", "IIb"): date(2027, 12, 31),
    ("AIMDD", "I"): date(2028, 12, 31),
    # IVDD — transition to IVDR 2017/746
    ("IVDD", "IV"): date(2025, 12, 31),   # Class D equivalent (highest risk)
    ("IVDD", "III"): date(2026, 12, 31),  # Class C equivalent
    ("IVDD", "II"): date(2027, 12, 31),   # Class B equivalent
    ("IVDD", "I"): date(2028, 12, 31),    # Class A equivalent
}

# Default deadline when class/cert_type combo is unknown
_DEFAULT_TRANSITION_DEADLINE = date(2028, 12, 31)

# MDD class → MDR equivalent class mapping
_MDD_TO_MDR_CLASS: dict[str, str] = {
    "I": "I",
    "I_sterile": "I",
    "I_measuring": "I",
    "IIa": "IIa",
    "IIb": "IIb",
    "III": "III",
}

# MDR articles applicable by device class (representative, not exhaustive)
_MDR_ARTICLES_BY_CLASS: dict[str, list[str]] = {
    "I": [
        "Article 10 (General obligations of manufacturers)",
        "Article 12 (Technical documentation — Annex II)",
        "Article 15 (Person responsible for regulatory compliance)",
        "Article 55 (Classification rules — Annex VIII)",
        "Article 61 (Clinical evaluation)",
        "Annex I (General safety and performance requirements)",
        "Annex II (Technical documentation)",
        "Annex IV (EU declaration of conformity)",
    ],
    "IIa": [
        "Article 10 (General obligations of manufacturers)",
        "Article 12 (Technical documentation — Annex II)",
        "Article 15 (Person responsible for regulatory compliance)",
        "Article 52 (Conformity assessment procedures)",
        "Article 55 (Classification rules — Annex VIII)",
        "Article 61 (Clinical evaluation)",
        "Article 83 (Post-market surveillance system)",
        "Article 86 (Periodic safety update report)",
        "Annex I (General safety and performance requirements)",
        "Annex II (Technical documentation)",
        "Annex IX (Conformity assessment — quality management)",
        "Annex XI (Product quality assurance)",
    ],
    "IIb": [
        "Article 10 (General obligations of manufacturers)",
        "Article 12 (Technical documentation — Annex II)",
        "Article 15 (Person responsible for regulatory compliance)",
        "Article 52 (Conformity assessment procedures)",
        "Article 55 (Classification rules — Annex VIII)",
        "Article 61 (Clinical evaluation)",
        "Article 74 (Clinical investigations)",
        "Article 83 (Post-market surveillance system)",
        "Article 86 (Periodic safety update report)",
        "Annex I (General safety and performance requirements)",
        "Annex II (Technical documentation)",
        "Annex IX (Conformity assessment — quality management)",
        "Annex X (Clinical evaluation — Annex X)",
    ],
    "III": [
        "Article 10 (General obligations of manufacturers)",
        "Article 12 (Technical documentation — Annex II)",
        "Article 15 (Person responsible for regulatory compliance)",
        "Article 52 (Conformity assessment procedures)",
        "Article 54 (Special procedure for Class III implantable and Class IIb)",
        "Article 55 (Classification rules — Annex VIII)",
        "Article 61 (Clinical evaluation)",
        "Article 74 (Clinical investigations)",
        "Article 83 (Post-market surveillance system)",
        "Article 85 (Post-market surveillance report)",
        "Article 86 (Periodic safety update report)",
        "Annex I (General safety and performance requirements)",
        "Annex II (Technical documentation)",
        "Annex III (Technical documentation on post-market surveillance)",
        "Annex IX (Conformity assessment — quality management)",
        "Annex X (Clinical evaluation — Annex X)",
        "Annex XI (Product quality assurance)",
    ],
}

# Known Notified Bodies (NB) code prefixes → org names
_NB_REGISTRY: dict[str, str] = {
    "BSI": "BSI Group (UK/DE — NB 0086)",
    "TUV": "TÜV SÜD Product Service (NB 0123)",
    "SGS": "SGS Belgium NV (NB 1639)",
    "DNV": "DNV (NB 0434)",
    "DEKRA": "DEKRA Certification (NB 0344)",
    "LNE": "Laboratoire National de Métrologie (NB 0050)",
    "GMED": "G-MED France (NB 0459)",
    "IMQ": "IMQ SpA Italy (NB 0051)",
    "NEMKO": "NEMKO AS Norway (NB 0366)",
    "INTERTEK": "Intertek (NB 1420)",
}

# Certificate number patterns for type detection
_CERT_PATTERNS = {
    "MDD": re.compile(
        r"(?:EC|MDD|93/42)[_\-/\s]?(?:\d{4})[_\-/\s]?(\w+)", re.IGNORECASE
    ),
    "AIMDD": re.compile(
        r"(?:AIMDD|90/385)[_\-/\s]?(?:\d{4})[_\-/\s]?(\w+)", re.IGNORECASE
    ),
    "IVDD": re.compile(
        r"(?:IVDD|98/79)[_\-/\s]?(?:\d{4})[_\-/\s]?(\w+)", re.IGNORECASE
    ),
    "MDR": re.compile(
        r"(?:MDR|2017/745)[_\-/\s]?(?:\d{4})[_\-/\s]?(\w+)", re.IGNORECASE
    ),
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LegacyMapping(BaseModel):
    """Result of mapping a legacy MDD/AIMDD/IVDD certificate to MDR."""

    mdd_cert: str = Field(..., description="Original legacy certificate number")
    cert_type: str = Field(
        default="MDD", description="Legacy directive type: MDD | AIMDD | IVDD"
    )
    mdd_class: str = Field(default="", description="Legacy device class (I/IIa/IIb/III)")
    mdr_equivalent_class: str = Field(
        default="", description="Equivalent MDR/IVDR device class"
    )
    applicable_mdr_articles: list[str] = Field(
        default_factory=list, description="MDR articles applicable to this device class"
    )
    transition_deadline: Optional[date] = Field(
        None, description="Final transition deadline per EU Reg 2024/1860"
    )
    is_still_valid: bool = Field(
        default=False, description="Whether the legacy certificate is still within transition period"
    )
    days_remaining: int = Field(
        default=0, description="Days until transition deadline (0 if expired)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Regulatory warnings for this certificate"
    )
    faiss_matched_requirements: list[str] = Field(
        default_factory=list,
        description="MDR requirements matched from FAISS vector store",
    )
    mapped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of this mapping operation",
    )
    notified_body: str = Field(default="", description="Issuing notified body")


class CertificateStatus(BaseModel):
    """Status check for a single legacy certificate."""

    cert_number: str
    is_valid: bool
    cert_type: str = ""
    mdd_class: str = ""
    days_remaining: int = 0
    transition_deadline: Optional[date] = None
    recommendation: str = ""


class PortfolioAudit(BaseModel):
    """Audit result for a portfolio of legacy certificates."""

    total_certificates: int
    still_valid: int
    expired: int
    expiring_soon: int = Field(description="Certificates expiring within 6 months")
    by_class: dict[str, int] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    certificates: list[CertificateStatus] = Field(default_factory=list)
    audited_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class LegacyMapper:
    """
    Maps legacy MDD/AIMDD/IVDD device certificates to MDR 2017/745 requirements.

    Uses FAISS vector store to find MDR requirements matching MDD classifications.
    Fully synchronous-safe: async methods exist for FAISS queries; all deadline
    computations are deterministic and do not require I/O.
    """

    def __init__(self) -> None:
        self._vector_store: Any = None

    def _get_vector_store(self) -> Any:
        if self._vector_store is None:
            from app.tools.vector_store import get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store

    # -- Certificate type detection -----------------------------------------

    @staticmethod
    def _detect_cert_type(cert_number: str) -> str:
        """Infer legacy directive type from certificate number format."""
        upper = cert_number.upper()
        if "AIMDD" in upper or "90/385" in upper:
            return "AIMDD"
        if "IVDD" in upper or "98/79" in upper:
            return "IVDD"
        if "MDR" in upper or "2017/745" in upper:
            return "MDR"
        # Default: MDD
        return "MDD"

    @staticmethod
    def _extract_class_from_cert(cert_number: str) -> str:
        """
        Attempt to extract device class from certificate number.

        Looks for Roman numerals (I, IIa, IIb, III) in the cert string.
        Returns empty string if ambiguous.
        """
        patterns = [
            (re.compile(r"\bIII\b", re.IGNORECASE), "III"),
            (re.compile(r"\bIIb\b", re.IGNORECASE), "IIb"),
            (re.compile(r"\bIIa\b", re.IGNORECASE), "IIa"),
            (re.compile(r"\bII\b", re.IGNORECASE), "IIa"),  # bare II assumed IIa
            (re.compile(r"\bI\b"), "I"),
        ]
        for pattern, cls in patterns:
            if pattern.search(cert_number):
                return cls
        return ""

    # -- Transition deadline logic ------------------------------------------

    def get_transition_deadline(
        self, device_class: str, cert_type: str = "MDD"
    ) -> date:
        """
        Return the transition deadline for a device class under the given legacy directive.

        Based on EU Regulation 2024/1860 amended transition schedule.
        Returns _DEFAULT_TRANSITION_DEADLINE if the combination is not found.
        """
        key = (cert_type.upper(), device_class.upper() if device_class else "I")

        # Normalise class: strip whitespace, handle mixed case
        normalised_class = device_class.strip() if device_class else "I"
        normalised_type = cert_type.strip().upper()

        # Try exact match first
        direct_key = (normalised_type, normalised_class)
        if direct_key in _TRANSITION_DEADLINES:
            return _TRANSITION_DEADLINES[direct_key]

        # Try uppercase normalisation
        for (t, c), dl in _TRANSITION_DEADLINES.items():
            if t == normalised_type and c.upper() == normalised_class.upper():
                return dl

        logger.debug(
            "legacy_mapper: no transition deadline for cert_type=%s class=%s, using default",
            cert_type,
            device_class,
        )
        return _DEFAULT_TRANSITION_DEADLINE

    # -- FAISS query for MDR requirements -----------------------------------

    async def search_mdr_requirements(
        self, mdd_classification: str, top_k: int = 8
    ) -> list[str]:
        """
        Use FAISS to find MDR requirement text matching an MDD classification.

        Searches the EU country partition of the vector store.
        Returns up to top_k human-readable requirement strings.
        """
        try:
            store = self._get_vector_store()
            query = f"EU MDR 2017/745 requirements for {mdd_classification} medical device"
            results = store.search(query=query, country="EU", top_k=top_k)
            return [
                f"{r.get('regulation_name', '')} {r.get('article', '')} — {r.get('text', '')[:120]}"
                for r in results
                if r.get("text")
            ]
        except Exception as exc:
            logger.debug("legacy_mapper FAISS search failed: %s", exc)
            return []

    # -- Core mapping -------------------------------------------------------

    async def map_mdd_to_mdr(
        self,
        mdd_cert: str,
        notified_body: str = "",
        device_class: str = "",
    ) -> LegacyMapping:
        """
        Map a legacy MDD/AIMDD/IVDD certificate to MDR 2017/745 requirements.

        Args:
            mdd_cert:       Legacy certificate number (e.g. "EC_2019_12345_ABC")
            notified_body:  Issuing NB name or code (optional, for display)
            device_class:   Override device class if known (I/IIa/IIb/III)

        Returns:
            LegacyMapping with MDR equivalence, transition deadline, and requirements.
        """
        cert_type = self._detect_cert_type(mdd_cert)
        inferred_class = device_class.strip() if device_class.strip() else self._extract_class_from_cert(mdd_cert)
        if not inferred_class:
            inferred_class = "IIb"  # Conservative default

        mdr_class = _MDD_TO_MDR_CLASS.get(inferred_class, inferred_class)
        applicable_articles = _MDR_ARTICLES_BY_CLASS.get(mdr_class, _MDR_ARTICLES_BY_CLASS["IIb"])

        deadline = self.get_transition_deadline(inferred_class, cert_type)
        today = date.today()
        days_remaining = max((deadline - today).days, 0)
        is_still_valid = today <= deadline

        # FAISS search for matched MDR requirements
        faiss_reqs = await self.search_mdr_requirements(
            f"Class {mdr_class} {cert_type} device"
        )

        # Build warnings
        warnings: list[str] = []
        if not is_still_valid:
            warnings.append(
                f"EXPIRED: Transition deadline {deadline} has passed. "
                "Device cannot be placed on EU market under legacy certificate."
            )
        elif days_remaining <= 180:
            warnings.append(
                f"EXPIRING SOON: {days_remaining} days until transition deadline {deadline}. "
                "Begin MDR conformity assessment immediately."
            )
        if cert_type == "AIMDD":
            warnings.append(
                "AIMDD devices are subject to Class III equivalent scrutiny under MDR. "
                "Notified body involvement mandatory (Annex IX/X)."
            )
        if mdr_class == "III":
            warnings.append(
                "Class III requires Annex IX Chapter II or Annex X conformity assessment. "
                "SCENIHR/SCHEER consultation may apply for certain implantables."
            )

        nb_display = ""
        if notified_body:
            nb_key = notified_body.upper()
            nb_display = next(
                (v for k, v in _NB_REGISTRY.items() if k in nb_key),
                notified_body,
            )

        return LegacyMapping(
            mdd_cert=mdd_cert,
            cert_type=cert_type,
            mdd_class=inferred_class,
            mdr_equivalent_class=mdr_class,
            applicable_mdr_articles=applicable_articles,
            transition_deadline=deadline,
            is_still_valid=is_still_valid,
            days_remaining=days_remaining,
            warnings=warnings,
            faiss_matched_requirements=faiss_reqs,
            notified_body=nb_display or notified_body,
            mapped_at=datetime.now(timezone.utc),
        )

    # -- Certificate status check -------------------------------------------

    async def check_certificate_validity(
        self, cert_number: str, device_class: str = ""
    ) -> CertificateStatus:
        """
        Check whether a legacy certificate is still within its transition window.

        Args:
            cert_number: Certificate identifier string
            device_class: Override device class if known

        Returns:
            CertificateStatus with validity, days remaining, and recommendation.
        """
        cert_type = self._detect_cert_type(cert_number)
        inferred_class = (
            device_class.strip()
            if device_class.strip()
            else self._extract_class_from_cert(cert_number) or "IIb"
        )

        deadline = self.get_transition_deadline(inferred_class, cert_type)
        today = date.today()
        days_remaining = max((deadline - today).days, 0)
        is_valid = today <= deadline

        if not is_valid:
            recommendation = (
                "Certificate has expired. Obtain MDR certificate before placing on EU market."
            )
        elif days_remaining <= 90:
            recommendation = (
                f"URGENT: {days_remaining} days remaining. Expedite MDR conformity assessment."
            )
        elif days_remaining <= 180:
            recommendation = (
                f"Initiate MDR conformity assessment within 30 days. "
                f"{days_remaining} days until deadline."
            )
        else:
            recommendation = (
                f"Certificate valid until {deadline} ({days_remaining} days). "
                "Plan MDR transition project."
            )

        return CertificateStatus(
            cert_number=cert_number,
            is_valid=is_valid,
            cert_type=cert_type,
            mdd_class=inferred_class,
            days_remaining=days_remaining,
            transition_deadline=deadline,
            recommendation=recommendation,
        )

    # -- Portfolio audit ----------------------------------------------------

    async def audit_legacy_portfolio(
        self, certificates: list[str]
    ) -> PortfolioAudit:
        """
        Audit a portfolio of legacy certificate numbers.

        Checks validity, groups by class, and generates prioritised recommendations.

        Args:
            certificates: List of legacy certificate number strings

        Returns:
            PortfolioAudit with summary counts and prioritised recommendations.
        """
        today = date.today()
        six_months = date(today.year + (1 if today.month > 6 else 0),
                          ((today.month + 6 - 1) % 12) + 1,
                          today.day)

        statuses: list[CertificateStatus] = []
        for cert in certificates:
            status = await self.check_certificate_validity(cert)
            statuses.append(status)

        still_valid = sum(1 for s in statuses if s.is_valid)
        expired = sum(1 for s in statuses if not s.is_valid)
        expiring_soon = sum(
            1 for s in statuses
            if s.is_valid and s.transition_deadline and s.transition_deadline <= six_months
        )

        by_class: dict[str, int] = {}
        for s in statuses:
            by_class[s.mdd_class] = by_class.get(s.mdd_class, 0) + 1

        recommendations: list[str] = []
        if expired > 0:
            recommendations.append(
                f"P0 CRITICAL: {expired} certificate(s) EXPIRED — remove from EU market immediately "
                "or obtain MDR certificates."
            )
        if expiring_soon > 0:
            recommendations.append(
                f"P1 HIGH: {expiring_soon} certificate(s) expire within 6 months — "
                "assign NB assessment project now."
            )
        class_iii_count = by_class.get("III", 0)
        if class_iii_count > 0:
            recommendations.append(
                f"P1 HIGH: {class_iii_count} Class III devices require Annex IX Chapter II or "
                "Annex X assessment — longest NB lead times."
            )
        if still_valid > 0:
            recommendations.append(
                f"{still_valid} certificate(s) still valid. "
                "Establish MDR transition roadmap with phased NB engagement."
            )

        return PortfolioAudit(
            total_certificates=len(certificates),
            still_valid=still_valid,
            expired=expired,
            expiring_soon=expiring_soon,
            by_class=by_class,
            recommendations=recommendations,
            certificates=statuses,
            audited_at=datetime.now(timezone.utc),
        )
