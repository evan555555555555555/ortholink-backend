"""
Wave 7: Gold content ingestion — local files from extracted zips.

Sources:
  1. AI Medical Device Guideline (Johner Institut) — EU MDR + AI Act compliance checklist
  2. RDM regulatory document templates — IEC 62304, ISO 14971, FDA SWP guidance
  3. openFDA device field schemas — MAUDE event, recall, clearance, PMA, UDI, enforcement
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import Chunk, chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store


def _read_file(path: str) -> str:
    """Read a local file, return its text content."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _chunk_local_doc(
    doc_id: str,
    title: str,
    text: str,
    country: str,
) -> list[Chunk]:
    """Chunk a local document using the regulatory chunker."""
    chunks = chunk_regulatory_text(
        text=text,
        country=country,
        regulation_name=title,
        document_id=doc_id,
    )
    return chunks


def _chunk_yaml_schema(
    doc_id: str,
    title: str,
    text: str,
    country: str,
) -> list[Chunk]:
    """Chunk a YAML schema into field-group chunks (not regulatory articles)."""
    import yaml
    import hashlib
    import uuid

    chunks: list[Chunk] = []

    try:
        data = yaml.safe_load(text)
    except Exception:
        # Fall back to text chunking
        return chunk_regulatory_text(
            text=text, country=country,
            regulation_name=title, document_id=doc_id,
        )

    if not isinstance(data, dict) or "properties" not in data:
        return chunk_regulatory_text(
            text=text, country=country,
            regulation_name=title, document_id=doc_id,
        )

    props = data["properties"]

    # Group fields into chunks of ~10 fields each
    field_items = list(props.items())
    group_size = 10

    for i in range(0, len(field_items), group_size):
        group = field_items[i : i + group_size]
        lines = []
        for field_name, field_info in group:
            desc = ""
            if isinstance(field_info, dict):
                desc = field_info.get("description", "")
                ftype = field_info.get("type", "")
                fmt = field_info.get("format", "")
                possible = field_info.get("possible_values", "")

                line = f"- {field_name}"
                if ftype:
                    line += f" ({ftype})"
                if fmt:
                    line += f" [format: {fmt}]"
                line += f": {desc}"
                if isinstance(possible, dict) and possible.get("value"):
                    vals = possible["value"]
                    if isinstance(vals, dict):
                        val_str = ", ".join(f"{k}={v}" for k, v in vals.items())
                        line += f" Values: {val_str}"
                lines.append(line)
            else:
                lines.append(f"- {field_name}: {field_info}")

        chunk_text = f"{title}\n\n" + "\n".join(lines)
        chunk_id = str(uuid.uuid4())

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            parent_text=chunk_text,
            country=country,
            regulation_name=title,
            article=f"Fields {i+1}-{i+len(group)}",
            document_id=doc_id,
        ))

    return chunks


def main():
    store = get_vector_store()
    all_chunks: list[Chunk] = []

    # ── 1. AI Medical Device Guideline ─────────────────────────────────────────
    ai_guide_path = "/tmp/ai-guideline-extract/ai-guideline-master/Guideline-AI-Medical-Devices_EN.md"
    if Path(ai_guide_path).exists():
        print("[wave7] Ingesting AI Medical Device Guideline (EU MDR + AI Act)...")
        text = _read_file(ai_guide_path)
        chunks = _chunk_local_doc(
            doc_id="AI_MEDICAL_DEVICE_GUIDELINE_EN",
            title="Guideline for AI-based Medical Devices and IVD (Johner Institut) — EU MDR 2017/745 + AI Act 2024/1689",
            text=text,
            country="EU",
        )
        print(f"  -> {len(chunks)} chunks from AI guideline")
        all_chunks.extend(chunks)
    else:
        print(f"[wave7] SKIP: {ai_guide_path} not found")

    # ── 2. RDM Regulatory Document Templates ──────────────────────────────────
    rdm_base = Path("/tmp/rdm-extract/rdm-main/rdm/init_files/documents")
    if rdm_base.exists():
        print("[wave7] Ingesting RDM regulatory document templates...")

        rdm_docs = [
            ("RDM_SOFTWARE_PLAN", "IEC 62304 Software Plan Template — Medical Device Software Lifecycle", "software_plan.md"),
            ("RDM_RISK_ASSESSMENT", "ISO 14971 Risk Assessment Template — Medical Device Risk Management", "risk_assessment.md"),
            ("RDM_RISK_MGMT_PLAN", "ISO 14971 Risk Management Plan Template — Medical Device Risk Management", "risk_management_plan.md"),
            ("RDM_RISK_MGMT_REPORT", "ISO 14971 Risk Management Report Template — Medical Device Risk Management", "risk_management_report.md"),
            ("RDM_SOFTWARE_DESC", "IEC 62304 Software Description Template — Device Software Functions", "software_description.md"),
            ("RDM_SOFTWARE_DESIGN", "IEC 62304 Software Design Specification Template", "software_design_specification.md"),
            ("RDM_SOFTWARE_REQ", "IEC 62304 Software Requirements Specification Template", "software_requirements_specification.md"),
            ("RDM_SOFTWARE_DEV_PRACTICES", "IEC 62304 Software Development and Maintenance Practices", "software_development_and_maintenance_practices.md"),
            ("RDM_VV_PLAN", "IEC 62304 Verification and Validation Plan Template", "verification_and_validation_plan.md"),
            ("RDM_SOFTWARE_RELEASE", "IEC 62304 Software Release Activity Record Template", "software_release_activity_record.md"),
            ("RDM_CYBER_SECURITY", "FDA Cybersecurity Guidance — Medical Device Cyber Security Documentation", "cyber_security.md"),
            ("RDM_DOC_LEVEL_EVAL", "IEC 62304 Documentation Level Evaluation — Safety Classification", "documentation_level_evaluation.md"),
            ("RDM_510K_OVERVIEW", "FDA 510(k) Submission Overview Template", "510k/00_overview.md"),
            ("RDM_510K_DECLARATIONS", "FDA 510(k) Declarations of Conformity and Summary Reports", "510k/09_declarations_of_conformity_and_summary_reports.md"),
            ("RDM_510K_EXECUTIVE_SUMMARY", "FDA 510(k) Executive Summary and Predicate Comparison", "510k/11_executive_summary_predicate_comparison.md"),
            ("RDM_510K_SUBSTANTIAL_EQ", "FDA 510(k) Substantial Equivalence Discussion Template", "510k/12_substantial_equivalence_discussion.md"),
        ]

        for doc_id, title, filename in rdm_docs:
            filepath = rdm_base / filename
            if filepath.exists():
                text = _read_file(str(filepath))
                # Strip Jinja2 template variables for cleaner chunks
                import re
                text = re.sub(r"\{\{[^}]+\}\}", "[DEVICE_SPECIFIC]", text)
                text = re.sub(r"\{%[^%]+%\}", "", text)
                text = re.sub(r"\[\[:.*?\]\]", "", text)

                chunks = _chunk_local_doc(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    country="STANDARDS",
                )
                print(f"  -> {len(chunks)} chunks from {filename}")
                all_chunks.extend(chunks)

        # Also ingest the risk.yml data model
        risk_yml = Path("/tmp/rdm-extract/rdm-main/rdm/init_files/data/risk.yml")
        if risk_yml.exists():
            text = _read_file(str(risk_yml))
            chunks = _chunk_local_doc(
                doc_id="RDM_RISK_MATRIX_DATA",
                title="ISO 14971 Risk Acceptability Matrix — Severity vs Probability Levels",
                text=text,
                country="STANDARDS",
            )
            print(f"  -> {len(chunks)} chunks from risk.yml")
            all_chunks.extend(chunks)
    else:
        print(f"[wave7] SKIP: {rdm_base} not found")

    # ── 3. openFDA Device Field Schemas ───────────────────────────────────────
    openfda_base = Path("/tmp/openfda-extract/open.fda.gov-master/src/constants/fields")
    if openfda_base.exists():
        print("[wave7] Ingesting openFDA device field schemas...")

        device_schemas = [
            ("OPENFDA_DEVICE_EVENT", "openFDA MAUDE Device Adverse Event Report — Field Schema", "deviceevent.yaml"),
            ("OPENFDA_DEVICE_RECALL", "openFDA Device Recall Enforcement — Field Schema", "devicerecall.yaml"),
            ("OPENFDA_DEVICE_CLASS", "openFDA Device Classification — Field Schema", "deviceclass.yaml"),
            ("OPENFDA_DEVICE_CLEARANCE", "openFDA 510(k) Device Clearance — Field Schema", "deviceclearance.yaml"),
            ("OPENFDA_DEVICE_PMA", "openFDA Premarket Approval (PMA) — Field Schema", "devicepma.yaml"),
            ("OPENFDA_DEVICE_UDI", "openFDA Unique Device Identification (UDI) — Field Schema", "deviceudi.yaml"),
            ("OPENFDA_DEVICE_ENFORCEMENT", "openFDA Device Enforcement Actions — Field Schema", "deviceenforcement.yaml"),
            ("OPENFDA_DEVICE_REGLIST", "openFDA Device Registration and Listing — Field Schema", "devicereglist.yaml"),
            ("OPENFDA_DEVICE_COVID19", "openFDA COVID-19 Serology Device Testing — Field Schema", "devicecovid19serology.yaml"),
        ]

        for doc_id, title, filename in device_schemas:
            filepath = openfda_base / filename
            if filepath.exists():
                text = _read_file(str(filepath))
                chunks = _chunk_yaml_schema(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    country="US",
                )
                print(f"  -> {len(chunks)} chunks from {filename}")
                all_chunks.extend(chunks)
    else:
        print(f"[wave7] SKIP: {openfda_base} not found")

    # ── Embed and index ───────────────────────────────────────────────────────
    if all_chunks:
        print(f"\n[wave7] Total chunks to embed: {len(all_chunks)}")
        count = embed_and_index_chunks(all_chunks, vector_store=store)
        print(f"[wave7] Successfully embedded and indexed: {count} chunks")
        total = len(store.metadata) if hasattr(store, 'metadata') else "unknown"
        print(f"[wave7] New total store size: {total}")
    else:
        print("[wave7] No chunks to process!")


if __name__ == "__main__":
    main()
