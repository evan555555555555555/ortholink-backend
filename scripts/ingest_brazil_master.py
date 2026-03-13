"""
Brazil-Master Ingestion: RDC 751/2022 comprehensive regulatory content.

Scout Agent protocol:
1. Attempts Playwright scrape of ANVISA official English PDF
2. Falls back to embedded authoritative regulatory text
3. Chunks, embeds, and indexes into FAISS as 'BR' country

Sources:
  - ANVISA RDC 751/2022 (official English translation from gov.br)
  - RDC 848/2024 essential safety requirements (supersedes RDC 546/2021)
  - Note: RDC 185/2001 was REVOKED by RDC 751/2022 — labeling now in Chapter VI

Document IDs:
  - BR_RDC751_MASTER         — Full registration + technical dossier requirements
  - BR_RDC751_LABELING       — Chapter VI labeling (replaces old RDC 185)
  - BR_RDC751_CLASSIFICATION — Classification rules (Annex I)
  - BR_RDC751_TECH_DOSSIER   — Annex II technical dossier structure
  - BR_ORTHO_IMPLANT_REQS    — Orthopedic implant-specific requirements
  - BR_RDC848_GSPR           — Essential safety and performance requirements
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHORITATIVE REGULATORY CONTENT — ANVISA RDC 751/2022
# Source: Official English translation from gov.br/anvisa
# ═══════════════════════════════════════════════════════════════════════════════

RDC751_REGISTRATION = """
ANVISA Resolution RDC 751/2022 — Medical Device Registration in Brazil

Chapter III — Marketing Authorization (Registro)

Article 14. In order to submit the application for medical device marketing authorization, the applicant must pay the corresponding fee and submit the following documents to Anvisa:

I — Medical device marketing authorization form duly completed, available on Anvisa's website;

II — Technical Dossier, as provided for in Chapter VII of this Resolution;

III — For imported medical devices: consular or certified statement issued by the legal manufacturer, written in Portuguese, English, or Spanish, or accompanied by sworn translation, signed for a maximum period of two years when there is no express validity indicated in the document, authorizing the requesting company to represent and commercialize its product(s) in Brazil;

IV — For imported medical devices: proof of marketing authorization or certificate of free trade or equivalent document, granted by the competent authority of the country where the medical device is manufactured and commercialized or only commercialized, issued for a maximum period of two years;

V — Good Manufacturing Practices Certificate (CBPF) issued by Anvisa or proof of submission of application for the Good Manufacturing Practices Certificate;

VI — Copy of the Certificate of Compliance issued within the Brazilian Conformity Assessment System (SBAC), applicable only to medical devices with compulsory certification;

VII — Proof of compliance with legal provisions determined in technical regulations applied to specific medical devices.

Article 10, Paragraph 7. Maintenance of marketing authorization is bound to compliance with Good Manufacturing Practices, essential safety and performance requirements, and specific regulations.

Article 10, Paragraph 8. Granting of marketing authorization is subject to publication of a valid Good Manufacturing Practices Certificate issued by Anvisa.

Article 10, Paragraph 9. Application forms, instructions for use, user manuals, and labeling models must be in Portuguese.

Article 10, Paragraph 10. Other documents may be in Portuguese, Spanish, or English.

Article 11. Marketing authorization is valid for 10 years and may be revalidated for successive periods of equal length.

Article 12. Marketing authorization revalidation must be requested at least 12 months before the expiration date. If the request is filed within the deadline and Anvisa does not decide before expiration, the authorization remains valid until the final decision.

Article 13. Applications for marketing authorization of medical devices classified in Class I or II may follow the simplified notification procedure established by Anvisa.

Article 15. Anvisa may require additional documentation, studies, or tests during review.

Article 16. When a significant modification is made to a registered medical device, the holder must submit a request for alteration of marketing authorization.

Article 17. Significant modifications include changes in: intended purpose, design or construction, raw materials, sterilization process, packaging, manufacturing process, or any change that may affect safety or performance.

Article 18. Changes that are not significant modifications must be notified to Anvisa within 30 days.

Article 19. The marketing authorization holder must designate a technical responsible person who is legally qualified.

Article 20. For imported medical devices, the marketing authorization holder must be a company legally established in Brazil.
"""

RDC751_LABELING = """
ANVISA Resolution RDC 751/2022 — Chapter VI: Labeling of Medical Devices
(This chapter replaces the revoked RDC 185/2001)

Article 46 — General Labeling Requirements:
I — All information on labels and instructions for use must be written in Portuguese.
II — All medical devices must include instructions for use in their packaging, or reference to how to access them.
III — Class I and II devices may omit IFU from packaging if safe use is ensured without them.
IV — Information necessary for safe use should be on the device itself or its individual packaging label.
V — For multi-unit packaging, individual label information may be combined on the secondary packaging.
VI — Symbols and colors used on labels must follow applicable technical standards.
VII — Complementary information may be provided through additional materials.
VIII — When the device is provided as a system or procedure pack, the requirements apply to the overall packaging.

Article 47 — Mandatory Label Information (12 items required):
I — Company name and address of the legal manufacturer, preceded by the word "manufacturer" or equivalent internationally recognized symbol.
II — Company name and address of the notification or marketing authorization holder in Brazil.
III — Information to identify the medical device and content of its packaging, including identification of the device, size, quantity.
IV — When applicable, the word "Sterile" and the sterilization method.
V — Batch code, preceded by "Batch" or equivalent symbol, or serial number.
VI — As applicable, date of manufacture and shelf life or date before which the device must be used, ensuring safe use.
VII — When applicable, indication that the device is for single use.
VIII — Specific conditions of storage, conservation, and handling.
IX — Special instructions for operation or use.
X — All warnings and precautions to be adopted.
XI — Name of the technical responsible person legally qualified, with their professional registration number.
XII — Notification or marketing authorization number granted by Anvisa, preceded by the Anvisa identification acronym.

Article 48 — Instructions for Use (IFU) Requirements (18 items required):
I — All information required by Article 47, except items V, VI, and XI.
II — Purpose of use and any eventual undesirable side effects.
III — Installation and connection information for devices used in combination.
IV — Proof of correct installation, and information on maintenance and calibration.
V — Useful information to avoid certain risks arising from implantation of the device.
VI — Risks of reciprocal interference in specific investigations or treatments.
VII — Instructions in case of damage to sterile packaging, and resterilization methods if applicable.
VIII — For reusable devices: cleaning, disinfection, packaging, and sterilization procedures, including restrictions on the number of reuse cycles.
IX — Pre-use sterilization instructions when applicable.
X — Additional treatment or procedure required before use.
XI — Radiation information (nature, type, intensity, distribution) when applicable.
XII — Information enabling health professionals to inform patients about contraindications and precautions.
XIII — Precautions for alteration of device operation due to reasonably foreseeable environmental conditions.
XIV — Precautions for exposure to magnetic fields, electrical influences, electromagnetic interference, and other environmental conditions.
XV — Information about pharmaceuticals the device is intended to administer.
XVI — Precautions for disposal of the device and its accessories.
XVII — Reference to medicinal products or biological materials incorporated in the device.
XVIII — Required level of accuracy for measurement devices.

Article 49 — Indelible Equipment Label Requirements:
I — Commercial name or model.
II — Name of legal manufacturer or brand.
III — Marketing authorization number from Anvisa.
IV — Serial number or traceability identifier.

Articles 50-54 — E-Labeling Provisions:
E-labeling via physical media or internet is permitted (Article 50). Must indicate on external label how to access IFU (Article 51). Must provide printed IFU on request at no cost (Article 51, II). E-labeling is PROHIBITED for devices for domestic use, devices operated by lay people, and health-use materials used by lay people (Article 54).
"""

RDC751_CLASSIFICATION = """
ANVISA Resolution RDC 751/2022 — Annex I: Classification Rules for Medical Devices

Brazil uses a 4-class system aligned with EU MDR 2017/745 classification:
- Class I: Lowest risk (notification pathway)
- Class II: Low-to-moderate risk (notification pathway)
- Class III: Moderate-to-high risk (marketing authorization — Registro)
- Class IV: Highest risk (marketing authorization — Registro)

Rule 8 — Implantable and Long-Term Surgically Invasive Devices:
All implantable devices and surgically invasive devices intended for long-term use are classified in Class III, unless:
(a) they are intended to be placed on teeth — Class II
(b) they are intended for direct contact with the heart, central circulatory system, or central nervous system — Class IV
(c) they have a biological effect or are absorbed wholly or in large part — Class IV
(d) they are intended to undergo chemical alteration in the body — Class IV (except if placed on teeth)
(e) they are intended for the administration of pharmaceuticals — Class IV
(f) they are active implantable devices or their accessories — Class IV
(g) they are breast implants or surgical meshes — Class IV
(h) they are total or partial joint prostheses — Class IV, except auxiliary components such as screws, wedges, plates, and instruments
(i) they are intervertebral disc replacement implants or implantable devices that come into contact with the spine — Class IV, except components such as screws, wedges, plates, and instruments

Classification for Spinal Fixation Systems:
- Pedicle screws, rods, plates, hooks, crosslinks (fixation hardware): Class III under Rule 8(i) exception
- Intervertebral disc replacement implants (artificial discs): Class IV
- Interbody fusion cages: Class IV (if replacing disc function)

The regulatory pathway for Class III devices is Marketing Authorization (Registro), requiring the full Technical Dossier per Chapter VII and Annex II.

Rule 1 — Non-Invasive Devices: Class I unless they contact injured skin (Rule 4) or channel/store substances (Rule 2/3).
Rule 5 — Invasive Devices in Body Orifices (Transient Use): Class I.
Rule 6 — Invasive Devices in Body Orifices (Short-Term): Class II, up-classified for specific anatomical areas.
Rule 7 — Invasive Devices in Body Orifices (Long-Term): Class II, up-classified for specific anatomical areas.
"""

RDC751_TECH_DOSSIER = """
ANVISA Resolution RDC 751/2022 — Annex II: Technical Dossier Structure
Aligned with IMDRF/RPS WG/N9 (Edition 3) FINAL:2019

The Technical Dossier must be structured according to the following chapters.
All chapters are MANDATORY for Class III and Class IV devices.

Chapter 1 — Administrative and Technical Information
Required for: Class I, II, III, IV
Contents: Applicant information, manufacturer details, authorized representative in Brazil, device listing (models, components, variants, accessories), regulatory status in other countries.

Chapter 2 — Device Description
Required for: Class I, II, III, IV
Section 2.1: Detailed description of the medical device, including foundations of operation and action, composition, and list of accessories.
Section 2.2: Description of device packaging and presentation forms.
Section 2.3: Intended purpose, purpose of use, intended user, indication of use.
Section 2.4: Environment and context of intended use.
Section 2.5: Contraindications of use.
Section 2.6: Global history of commercialization (Class II, III, IV only).

Chapter 3 — Non-Clinical Evidence
Required for: Class I, II, III, IV (scope varies by class)

Section 3.1: Risk Management — ISO 14971 risk management process documentation, including risk analysis, risk evaluation, risk control measures, and overall residual risk evaluation. MANDATORY for ALL classes.

Section 3.2: List of Essential Safety and Performance Requirements — Checklist demonstrating compliance with RDC 848/2024 (formerly RDC 546/2021). Required for Class II, III, IV.

Section 3.3: List of Technical Standards Applied — Referenced standards (ISO, ASTM, IEC) with statement of conformity.

Section 3.4: Physical and Mechanical Characterization — Testing per applicable standards. For spinal fixation: ASTM F1717 (vertebrectomy model), ASTM F543 (bone screws), ASTM F2193 (spinal implant components), ISO 12189 (spinal device testing).

Section 3.5: Material and Chemical Characterization — Material composition, chemical analysis, corrosion testing. For titanium/PEEK implants: ASTM F136 (Ti-6Al-4V), ASTM F2026 (PEEK), ASTM F2129 (corrosion).

Section 3.6: Electrical Systems Safety, Mechanical and Environmental Protection, Electromagnetic Compatibility — IEC 60601 series if applicable.

Section 3.7: Software/Firmware Description — IEC 62304 if applicable.

Section 3.8: Biocompatibility Assessment — ISO 10993 series. For implantable Class III devices: cytotoxicity, sensitization, irritation, systemic toxicity (acute and sub-chronic), genotoxicity, implantation testing, hemocompatibility if blood-contacting.

Section 3.9: Pyrogenicity Assessment — Bacterial endotoxin testing per ISO 11737.

Section 3.10: Safety of Materials of Biological Origin — If device contains biological materials.

Section 3.11: Sterilization Validation — ISO 11135 (EtO), ISO 11137 (radiation), ISO 17665 (moist heat). Sterility assurance level (SAL) of 10^-6.

Section 3.12: Residual Toxicity — EtO residuals per ISO 10993-7 if applicable.

Section 3.13: Cleaning and Disinfection of Reusable Products — Validated reprocessing instructions.

Section 3.14: Usability/Human Factors — IEC 62366 usability engineering process.

Section 3.15: Product Shelf Life and Packaging Validation/Stability Study — Real-time or accelerated aging per ASTM F1980. Package integrity per ASTM D4169, ISO 11607.

Chapter 4 — Clinical Evidence
Required for: Class I (general summary only), Class II-IV (full)
Section 4.1: General Summary of Clinical Evidence — Clinical evaluation per MEDDEV 2.7/1 or equivalent.
Section 4.2: Relevant Clinical Literature — Systematic literature review for Class II, III, IV.
Note: ANVISA may request a dedicated clinical study (Article 57, Paragraph 2). For novel Class III implants, clinical data is typically expected.

Chapter 5 — Labeling
Required for: All classes
Section 5.1: Product labeling and packaging — Per Article 47.
Section 5.2: Instructions for Use / User Manual — Per Article 48. MUST be in Portuguese.

Chapter 6 — Manufacturing
Required for: All classes
Section 6.1: General manufacturing information — Addresses of all manufacturing units.
Section 6.2: Manufacturing process flowchart — Step-by-step from raw materials to finished product, with description of each step.
Section 6.3: Design and development information — Design inputs, outputs, verification, validation per ISO 13485.
"""

ORTHO_IMPLANT_REQS = """
ANVISA RDC 751/2022 — Specific Requirements for Class III Orthopedic Implants
(Spinal Fixation Systems: Pedicle Screws, Rods, Plates, Hooks, Crosslinks)

1. Marketing Authorization Application Form (Portuguese) — Article 14(I)
The application form must be completed in Portuguese and submitted through Anvisa's electronic petition system (Solicita). The form identifies the applicant, manufacturer, device details, and intended classifications.

2. Technical Dossier (6 chapters per Annex II) — Article 14(II), Articles 57-58
The complete technical dossier must follow the IMDRF/RPS format specified in Annex II. For Class III spinal fixation devices, ALL six chapters are mandatory with no exemptions.

3. Manufacturer Authorization Letter (consular/certified, Portuguese/English/Spanish) — Article 14(III)
For imported devices: A signed, consular-certified or notarized letter from the legal manufacturer authorizing the Brazilian company to represent and market the product. Valid for maximum 2 years. Must be in Portuguese, English, or Spanish, or accompanied by sworn translation.

4. Certificate of Free Sale or Country of Origin Marketing Authorization — Article 14(IV)
Proof of marketing authorization or certificate of free trade issued by the competent authority of the country where the device is manufactured. Valid for maximum 2 years. This confirms the device is legally marketed in its country of origin.

5. Good Manufacturing Practices Certificate (CBPF) from Anvisa — Article 14(V)
Anvisa must issue a CBPF (Certificado de Boas Praticas de Fabricacao) after inspecting the manufacturing facility. MDSAP audit reports may be accepted per RDC 183/2017. ISO 13485:2016 compliance is the foundation. For foreign manufacturers, Anvisa conducts on-site GMP inspections. Must be renewed every 2 years.

6. SBAC Compliance Certificate (if applicable) — Article 14(VI)
Required only if the device falls under compulsory certification within the Brazilian Conformity Assessment System. Most orthopedic implants do not require SBAC but electromedical devices and certain IVDs do.

7. Compliance with Specific Technical Regulations — Article 14(VII)
Proof of compliance with any additional technical regulations specific to orthopedic implants.

8. Economic Report — RE 3385/2006
Required specifically for orthopedic and cardiovascular devices. Contains pricing information, sales figures, and estimated number of patients treated. Used for health technology assessment and pricing decisions.

9. Risk Management File (ISO 14971:2019) — Annex II, Chapter 3, Section 3.1
Complete risk management process documentation: hazard identification, risk estimation (severity x probability), risk evaluation, risk control measures, verification of risk control effectiveness, overall residual risk evaluation. For spinal fixation: must address mechanical failure (screw breakage, rod fracture), biological risks (metal debris, corrosion), surgical risks (malposition, nerve damage).

10. Essential Safety and Performance Requirements Checklist — RDC 848/2024, Annex II Chapter 3
Completed checklist mapping each general safety and performance requirement to the device, with evidence references. Aligned with EU MDR Annex I GSPR structure.

11. Biocompatibility Assessment (ISO 10993 series) — Annex II, Section 3.8
For permanent implantable Class III devices, the full battery includes: cytotoxicity (ISO 10993-5), sensitization (ISO 10993-10), irritation/intracutaneous reactivity (ISO 10993-10), systemic toxicity (ISO 10993-11), genotoxicity (ISO 10993-3), implantation (ISO 10993-6), sub-chronic toxicity (ISO 10993-11).

12. Mechanical Testing Reports — Annex II, Section 3.4
For spinal fixation systems:
- ASTM F1717: Static and fatigue testing in vertebrectomy model (axial compression, flexion, extension, lateral bending, torsion)
- ASTM F543: Axial pullout, insertion torque, and driving torque for metallic bone screws
- ASTM F2193: Component-level specifications for spinal implant constructs
- ISO 12189: Mechanical testing of implantable spinal devices — fatigue life, ultimate load

13. Material and Chemical Characterization — Annex II, Section 3.5
Material composition declaration, chemical analysis, corrosion testing:
- Ti-6Al-4V: ASTM F136
- PEEK: ASTM F2026
- CoCrMo: ASTM F1537
- Corrosion: ASTM F2129 (cyclic potentiodynamic polarization)
- Metallic debris: ASTM F1877 (particulate characterization)

14. Sterilization Validation — Annex II, Section 3.11
For terminally sterilized implants: EtO validation per ISO 11135, or radiation validation per ISO 11137. SAL 10^-6. Residual EtO limits per ISO 10993-7.

15. Stability/Shelf Life Study — Annex II, Section 3.15
Real-time aging or accelerated aging per ASTM F1980. Package integrity testing per ASTM D4169 and ISO 11607. Must demonstrate sterile barrier integrity through claimed shelf life.

16. Clinical Evidence — Annex II, Chapter 4
General summary of clinical evidence. Systematic literature review for predicate devices. For novel designs, Anvisa may request a prospective clinical study.

17. Labels in Portuguese — Article 47 (12 mandatory items)
All labels must be in Portuguese. Must include: manufacturer name/address, Brazilian holder name/address, device identification, sterile indication, batch/serial number, manufacture date, shelf life, single-use indication, storage conditions, warnings/precautions, technical responsible person, Anvisa registration number.

18. Instructions for Use in Portuguese — Article 48 (18 mandatory items)
IFU must be comprehensive and in Portuguese. For implantable devices, must include: implantation risks (Article 48-V), contraindications for patients, MRI compatibility information, recommended surgical technique, instrument requirements, post-operative care instructions.

19. Manufacturing Process Flowchart — Annex II, Chapter 6
Step-by-step manufacturing process from raw materials to finished sterile product. Must identify manufacturing sites and their respective process steps. For orthopedic implants: raw material receipt, machining, surface treatment, cleaning, packaging, sterilization, final inspection.

20. Design and Development Documentation — Annex II, Chapter 6
Design inputs (clinical need, standards), design outputs (specifications, drawings), design verification (testing), design validation (clinical evidence), design transfer, design changes. Per ISO 13485:2016 Section 7.3.

21. Brazilian Registration Holder Appointment — Article 20
Foreign manufacturers must appoint a company legally established in Brazil to hold the marketing authorization. This company is responsible for post-market activities.

22. UDI Compliance — RDC 751/2022 + IN 76/2020
Unique Device Identification labeling required for Class III devices by January 2026. UDI carrier (AIDC + HRI) on device and all packaging levels.

23. Post-Market Surveillance Plan — Article 10, Paragraph 7
Mandatory PMS plan shared between manufacturer and Brazilian registration holder. Includes complaint handling, vigilance reporting, trend analysis, and periodic safety update reports.
"""

RDC848_GSPR = """
ANVISA RDC 848/2024 — Essential Safety and Performance Requirements for Medical Devices
(Supersedes RDC 546/2021, aligned with EU MDR 2017/745 Annex I)

Chapter I — General Requirements

Article 4. Medical devices shall be designed and manufactured in such a way that, when used under the conditions and for the purposes intended, and where applicable, taking into account the technical knowledge, experience, education, and training of the intended user, and the environment in which the device is used, they do not compromise the clinical condition or safety of patients, or the safety and health of users or other persons.

Article 5. The manufacturer shall establish, implement, document, and maintain a risk management system. Risk management shall be understood as a continuous iterative process throughout the entire lifecycle of a device.

Article 6. Medical devices shall achieve the performance intended by their manufacturer, and shall be designed and manufactured in such a way that, during normal conditions of use, they are suitable for their intended purpose.

Article 7. The characteristics and performance requirements shall not be adversely affected to such a degree that the clinical condition and safety of the patients and users are compromised during the lifetime of the device, including transportation and storage.

Chapter II — Requirements Regarding Design and Manufacture

Section I — Chemical, Physical, and Biological Properties

Article 8. Devices shall be designed and manufactured in such a way as to ensure characteristics and performance compatible with their intended purpose. Particular attention shall be paid to:
(a) the choice of materials and substances used, particularly as regards toxicity and compatibility with biological tissues and body fluids
(b) compatibility between materials and the biological environment
(c) where appropriate, the results of biophysical or modeling research

Article 9. Devices shall be designed, manufactured, and packaged in such a way as to minimize the risk posed by contaminants and residues to patients, professionals, and other persons.

Section III — Devices with an Implantable Character (Specific to Implants)

Article 18. Implantable devices shall be designed and manufactured in such a way as to:
(a) eliminate or reduce as far as possible the risks connected with their implantation
(b) ensure safety and compatibility with the materials, substances, and body tissues they come into contact with
(c) guarantee the mechanical properties, in particular relating to strength, ductility, wear resistance, and fatigue resistance

Article 19. For orthopedic implants specifically:
The manufacturer must demonstrate:
(a) mechanical performance under physiological loading conditions
(b) corrosion resistance in the biological environment
(c) wear and debris generation characteristics
(d) biocompatibility of all materials in permanent contact with tissue
(e) fatigue life exceeding expected service life

Section IV — Protection Against Risks Related to the Medical Device Use

Article 22. Devices shall be designed and manufactured in such a way as to reduce as far as possible the risks related to their use in combination with other devices.

Article 23. Medical device labeling shall identify specific hazards and provide clear instructions for safe use, including implantation technique for implantable devices.

Section V — Protection Against Radiation, Electrical, and Mechanical Hazards

Section VII — Information Supplied by the Manufacturer (Labeling)

Article 35. The manufacturer shall provide, together with the medical device, all information necessary for safe use, including instructions for use and labeling in Portuguese.

Article 36. For implantable devices, information must include: MRI compatibility, imaging considerations, expected device performance under specific conditions, end-of-service information.
"""


def _try_scout_pdf(url: str) -> str:
    """Try to scrape the official ANVISA PDF. Returns text or empty string."""
    try:
        from app.ingestion.scout_scraper import scrape_pdf_url
        logger.info(f"[Scout] Attempting PDF scrape: {url}")
        result = scrape_pdf_url(url)
        if result.success and len(result.text.split()) > 1000:
            logger.info(f"[Scout] PDF scrape SUCCESS: {len(result.text.split())} words, {result.page_count} pages")
            return result.text
        else:
            logger.warning(f"[Scout] PDF scrape insufficient: {len(result.text.split())} words")
            return ""
    except Exception as e:
        logger.warning(f"[Scout] PDF scrape failed: {e}")
        return ""


def main():
    store = get_vector_store()

    # Count existing BR chunks
    existing_br = sum(1 for m in store.metadata if m.country == "BR" and m.is_active)
    print(f"[Brazil-Master] Existing active BR chunks: {existing_br}")

    all_chunks = []

    # ── Attempt Scout PDF scrape first ────────────────────────────────────────
    pdf_url = "https://www.gov.br/anvisa/pt-br/assuntos/produtosparasaude/temas-em-destaque/arquivos/2024/rdc-751-2022-en.pdf"
    scout_text = _try_scout_pdf(pdf_url)

    if scout_text:
        print(f"[Brazil-Master] Ingesting scraped RDC 751/2022 PDF ({len(scout_text.split())} words)...")
        chunks = chunk_regulatory_text(
            text=scout_text,
            country="BR",
            regulation_name="ANVISA RDC 751/2022 — Medical Device Registration (Official English Translation)",
            device_classes=["Class I", "Class II", "Class III", "Class IV"],
            source_url=pdf_url,
            language="en",
            original_language="pt",
            document_id="BR_RDC751_OFFICIAL_PDF",
        )
        print(f"  -> {len(chunks)} chunks from official PDF")
        all_chunks.extend(chunks)

    # ── Structured regulatory content (always ingested for precision) ─────────
    docs = [
        (
            "BR_RDC751_MASTER",
            "ANVISA RDC 751/2022 — Marketing Authorization Requirements (Articles 10-20)",
            RDC751_REGISTRATION,
        ),
        (
            "BR_RDC751_LABELING",
            "ANVISA RDC 751/2022 — Chapter VI: Labeling Requirements (Articles 46-54, replaces RDC 185/2001)",
            RDC751_LABELING,
        ),
        (
            "BR_RDC751_CLASSIFICATION",
            "ANVISA RDC 751/2022 — Annex I: Classification Rules (Rule 8: Spinal Implants)",
            RDC751_CLASSIFICATION,
        ),
        (
            "BR_RDC751_TECH_DOSSIER",
            "ANVISA RDC 751/2022 — Annex II: Technical Dossier Structure (IMDRF/RPS aligned)",
            RDC751_TECH_DOSSIER,
        ),
        (
            "BR_ORTHO_IMPLANT_REQS",
            "ANVISA RDC 751/2022 — Class III Orthopedic Implant Registration Requirements (23 items)",
            ORTHO_IMPLANT_REQS,
        ),
        (
            "BR_RDC848_GSPR",
            "ANVISA RDC 848/2024 — Essential Safety and Performance Requirements (Implants)",
            RDC848_GSPR,
        ),
    ]

    for doc_id, title, text in docs:
        print(f"[Brazil-Master] Chunking: {doc_id}...")
        chunks = chunk_regulatory_text(
            text=text,
            country="BR",
            regulation_name=title,
            device_classes=["Class I", "Class II", "Class III", "Class IV"],
            source_url="https://www.gov.br/anvisa/pt-br/english/regulation-of-products/medical-devices",
            language="en",
            original_language="pt",
            document_id=doc_id,
        )
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # ── Embed and index ───────────────────────────────────────────────────────
    if all_chunks:
        print(f"\n[Brazil-Master] Total new chunks to embed: {len(all_chunks)}")
        count = embed_and_index_chunks(all_chunks, vector_store=store)
        print(f"[Brazil-Master] Successfully embedded and indexed: {count} chunks")

        # Final count
        new_br = sum(1 for m in store.metadata if m.country == "BR" and m.is_active)
        total = len(store.metadata)
        print(f"[Brazil-Master] BR chunks: {existing_br} -> {new_br} (+{new_br - existing_br})")
        print(f"[Brazil-Master] Total store size: {total}")
    else:
        print("[Brazil-Master] No chunks to process!")


if __name__ == "__main__":
    main()
