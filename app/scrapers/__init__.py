"""
OrthoLink Scrapers Package

Enforcement / vigilance data scrapers for all supported regulatory jurisdictions.

Available scrapers:
  - FDAWarningLettersScraper  (fda_warning_letters)  — US FDA warning letters + enforcement
  - TGAAlertsScraper          (tga_alerts)            — AU TGA safety alerts + device recalls
  - EUDAMEDFSNScraper         (eudamed_fsn)           — EU EUDAMED field safety notices
  - HCIncidentsScraper        (hc_incidents)          — CA Health Canada incident reports
  - MarketSurveillanceScraper (market_surveillance)   — EU EUDAMED market surveillance actions

All scrapers inherit from BaseEnforcementScraper and return EnforcementAction instances.
Singleton instances are exported for convenience.
"""

from app.scrapers.base_scraper import (
    BaseEnforcementScraper,
    EnforcementAction,
    Severity,
)
from app.scrapers.eudamed_fsn import (
    DeviceRiskProfile,
    EUDAMEDFSNScraper,
    FieldSafetyNotice,
    eudamed_fsn_scraper,
)
from app.scrapers.fda_warning_letters import (
    CitationPattern,
    FDAWarningLettersScraper,
    WarningLetter,
    fda_wl_scraper,
)
from app.scrapers.hc_incidents import (
    HCIncidentReport,
    HCIncidentsScraper,
    hc_incidents_scraper,
)
from app.scrapers.market_surveillance import (
    CountrySurveillanceSummary,
    MarketSurveillanceAction,
    MarketSurveillanceScraper,
    market_surveillance_scraper,
)
from app.scrapers.tga_alerts import (
    TGAAlertsScraper,
    TGASafetyAlert,
    tga_alerts_scraper,
)

__all__ = [
    # Base
    "BaseEnforcementScraper",
    "EnforcementAction",
    "Severity",
    # FDA
    "FDAWarningLettersScraper",
    "WarningLetter",
    "CitationPattern",
    "fda_wl_scraper",
    # TGA
    "TGAAlertsScraper",
    "TGASafetyAlert",
    "tga_alerts_scraper",
    # EUDAMED FSN
    "EUDAMEDFSNScraper",
    "FieldSafetyNotice",
    "DeviceRiskProfile",
    "eudamed_fsn_scraper",
    # Health Canada
    "HCIncidentsScraper",
    "HCIncidentReport",
    "hc_incidents_scraper",
    # EUDAMED Market Surveillance
    "MarketSurveillanceScraper",
    "MarketSurveillanceAction",
    "CountrySurveillanceSummary",
    "market_surveillance_scraper",
]
