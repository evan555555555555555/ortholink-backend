#!/usr/bin/env python3
"""
Deactivate all CFS- synthetic chunks in FAISS metadata.

Sets is_active=False on every chunk whose chunk_id starts with "CFS-".
Does NOT delete chunks (respects HC-6 no-delete rule) — only marks inactive
so they are excluded from search results.

Usage:
    python scripts/deactivate_cfs_seeds.py
"""

import json
import sys
from pathlib import Path

METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "metadata.json"


def main():
    if not METADATA_PATH.exists():
        print(f"ERROR: metadata.json not found at {METADATA_PATH}")
        sys.exit(1)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    total = len(metadata)
    deactivated = 0

    for entry in metadata:
        cid = entry.get("chunk_id", "")
        if cid.startswith("CFS-") and entry.get("is_active", True):
            entry["is_active"] = False
            deactivated += 1

    if deactivated == 0:
        print(f"No active CFS- chunks found in {total} entries. Nothing to do.")
        return

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Deactivated {deactivated} CFS- synthetic chunks out of {total} total entries.")
    print(f"Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
