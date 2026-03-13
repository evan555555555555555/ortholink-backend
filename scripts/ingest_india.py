#!/usr/bin/env python3
"""
OrthoLink India Ingestion (A4)
CDSCO Medical Devices Rules 2017 (PDF).
document_id: IN-MDR-2017.
device_classes: map CDSCO Class A→I, B→II, C→IIb, D→III (all for full doc).
Run: poetry run python scripts/ingest_india.py --file path/to/Medical-Devices-Rules-2017.pdf
Then: poetry run python -m app.ingestion.chunk_audit --country IN --sample 10
"""

import importlib.util
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_spec = importlib.util.spec_from_file_location(
    "ingest_country",
    Path(__file__).parent / "ingest_country.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ingest = _mod.ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ingest_india")

# CDSCO Class A→I, B→II, C→IIb, D→III — ingest with all for baseline
DEVICE_CLASSES_IN = ["I", "II", "IIb", "III"]


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Ingest India CDSCO Medical Devices Rules 2017 (PDF)")
    p.add_argument("--file", required=True, help="Path to Medical-Devices-Rules-2017.pdf")
    args = p.parse_args()

    try:
        ingest(
            country="IN",
            regulation_name="Medical Devices Rules 2017",
            file_path=args.file,
            device_classes=DEVICE_CLASSES_IN,
            language="en",
            document_id="IN-MDR-2017",
            skip_translation=True,
            skip_audit=False,
            embed_batch_size=10,
        )
    except Exception as e:
        logger.exception(e)
        return 1
    logger.info("India ingestion complete. Run: python -m app.ingestion.chunk_audit --country IN --sample 10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
