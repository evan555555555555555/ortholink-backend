"""
OrthoLink --- Registry Adapters

Data source adapters for international medical device regulatory databases.
Each adapter inherits from BaseRegistryAdapter and provides async access
to a specific country/global registry.

Registries:
  GUDID      -- USA: NLM AccessGUDID (FDA device identifiers)
  EUDAMED    -- EU: European Database on Medical Devices (MDR/IVDR)
  SWISSDAMED -- CH: Swissmedic Swissdamed FHIR R4
  ANVISA     -- BR: ANVISA SIUD (Normative Instruction 426/2026)
  ARTG       -- AU: TGA Australian Register of Therapeutic Goods
  MDALL      -- CA: Health Canada Medical Device Active Licence Listing
  SUGAM      -- IN: CDSCO SUGAM (Medical Devices Rules 2017)
  GMDN       -- Global: GMDN Agency nomenclature and cross-references
"""

from app.adapters.base_adapter import (
    AdapterHealthStatus,
    BaseRegistryAdapter,
    DeviceRecord,
    RegistrationRecord,
)
from app.adapters.gudid_adapter import GUDIDAdapter, GUDIDDevice
from app.adapters.eudamed_adapter import (
    EUDAMEDAdapter,
    EUDAMEDDevice,
    EUDAMEDCertificate,
    EUDAMEDFieldSafetyNotice,
)
from app.adapters.swissdamed_adapter import SwissdamedAdapter, SwissdamedDevice
from app.adapters.anvisa_adapter import ANVISAAdapter, ANVISADevice
from app.adapters.artg_adapter import ARTGAdapter, ARTGDevice
from app.adapters.mdall_adapter import MDALLAdapter, MDALLDevice, MDELEstablishment
from app.adapters.sugam_adapter import SUGAMAdapter, SUGAMDevice
from app.adapters.gmdn_adapter import GMDNAdapter, GMDNTerm

__all__ = [
    # Base
    "BaseRegistryAdapter",
    "DeviceRecord",
    "RegistrationRecord",
    "AdapterHealthStatus",
    # GUDID (USA)
    "GUDIDAdapter",
    "GUDIDDevice",
    # EUDAMED (EU)
    "EUDAMEDAdapter",
    "EUDAMEDDevice",
    "EUDAMEDCertificate",
    "EUDAMEDFieldSafetyNotice",
    # Swissdamed (CH)
    "SwissdamedAdapter",
    "SwissdamedDevice",
    # ANVISA (BR)
    "ANVISAAdapter",
    "ANVISADevice",
    # ARTG (AU)
    "ARTGAdapter",
    "ARTGDevice",
    # MDALL (CA)
    "MDALLAdapter",
    "MDALLDevice",
    "MDELEstablishment",
    # SUGAM (IN)
    "SUGAMAdapter",
    "SUGAMDevice",
    # GMDN (Global)
    "GMDNAdapter",
    "GMDNTerm",
]
