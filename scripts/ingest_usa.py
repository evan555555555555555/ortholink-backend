#!/usr/bin/env python3
"""
OrthoLink USA Ingestion (A3)
Ingest 21 CFR Part 801 (Labeling), 803 (MDR), 806 (Recalls), 820 (QMSR).
eCFR English — no translation. Chunk at § section boundaries.
Run: poetry run python scripts/ingest_usa.py
Then: poetry run python -m app.ingestion.chunk_audit --country US --sample 10
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
logger = logging.getLogger("ingest_usa")

USA_PARTS = [
    ("801", "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-801", "21 CFR Part 801", ["I", "II", "III"]),
    ("803", "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-803", "21 CFR Part 803", ["I", "II", "III"]),
    ("806", "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-806", "21 CFR Part 806", ["I", "II", "III"]),
    ("820", "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820", "21 CFR Part 820", ["II", "III"]),
]


def main() -> int:
    for part_num, url, regulation_name, device_classes in USA_PARTS:
        try:
            ingest(
                country="US",
                regulation_name=regulation_name,
                source_url=url,
                device_classes=device_classes,
                language="en",
                document_id=f"US-21CFR-{part_num}",
                skip_translation=True,
                skip_audit=True,
            )
        except Exception as e:
            logger.exception(e)
            return 1

    logger.info("USA ingestion complete. Run: python -m app.ingestion.chunk_audit --country US --sample 10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
