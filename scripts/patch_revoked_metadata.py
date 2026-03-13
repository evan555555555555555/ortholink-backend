"""
Fix-2: Metadata Lifecycle Patch Script
=======================================
Tags all currently-active FAISS chunks with `regulatory_status = "ACTIVE"` where
the field is currently None/missing.

Special attention to:
- BR chunks: Marks all active BR chunks as "ACTIVE"; identifies any that
  reference RDC 185/2001 (revoked 2022) in their text for reporting.
- Future use: To tag a chunk as REVOKED, set chunk["regulatory_status"] = "REVOKED"
  manually and re-run; the FAISS pre-filter will exclude it from all searches.

Usage:
    cd backend && source .venv/bin/activate
    python scripts/patch_revoked_metadata.py [--dry-run]

Outputs:
    - Patches metadata.json in-place (unless --dry-run)
    - Reports count of chunks tagged/skipped
"""

import json
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

METADATA_PATH = Path(__file__).parent.parent / "data" / "embeddings" / "metadata.json"

# Revocation registry: maps regulation patterns → replacement info
# Used for reporting only (not for automated tagging of existing chunks)
REVOCATION_REGISTRY = {
    "RDC 185": {
        "revoked_by": "ANVISA RDC 751/2022",
        "effective_date": "2022-08-01",
        "note": "RDC 185/2001 revoked; all labeling requirements now under RDC 751/2022 Chapter VI",
    },
    "RDC 56": {
        "revoked_by": "ANVISA RDC 751/2022",
        "effective_date": "2022-08-01",
        "note": "RDC 56/2001 revoked; consolidated into RDC 751/2022",
    },
}


def main():
    print(f"[patch_revoked_metadata] Loading: {METADATA_PATH}")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    total = len(metadata)
    tagged_active = 0
    already_had_status = 0
    revoked_refs_found: list[dict] = []

    for chunk in metadata:
        current_status = chunk.get("regulatory_status")
        is_active = chunk.get("is_active", True)
        regulation_name = chunk.get("regulation_name", "") or ""
        chunk_text = chunk.get("text", "") or ""
        country = chunk.get("country", "")

        # Track chunks already with explicit status
        if current_status is not None:
            already_had_status += 1
            continue

        # Tag all active chunks as ACTIVE (explicit lifecycle status)
        if is_active:
            if not DRY_RUN:
                chunk["regulatory_status"] = "ACTIVE"
            tagged_active += 1

        # Identify chunks that reference revoked regulations (for reporting)
        for revoked_pattern, info in REVOCATION_REGISTRY.items():
            if (
                revoked_pattern in regulation_name
                or revoked_pattern in chunk_text
            ):
                revoked_refs_found.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "country": country,
                    "regulation_name": regulation_name,
                    "revoked_ref": revoked_pattern,
                    "replacement": info["revoked_by"],
                    "note": info["note"],
                })

    # Ensure superseded_by field exists on all chunks (Fix-2 backward compat)
    missing_superseded_by = 0
    for chunk in metadata:
        if "superseded_by" not in chunk:
            chunk["superseded_by"] = None
            missing_superseded_by += 1

    print(f"\n[patch_revoked_metadata] Summary:")
    print(f"  Total chunks: {total}")
    print(f"  Tagged ACTIVE (new): {tagged_active}")
    print(f"  Already had status: {already_had_status}")
    print(f"  Added superseded_by=None: {missing_superseded_by}")
    print(f"  Revoked-regulation references found: {len(revoked_refs_found)}")

    if revoked_refs_found:
        print("\n[patch_revoked_metadata] Chunks referencing revoked regulations:")
        for ref in revoked_refs_found[:10]:
            print(f"  [{ref['country']}] {ref['chunk_id'][:16]}... "
                  f"{ref['regulation_name'][:60]} → ref: {ref['revoked_ref']}")
        if len(revoked_refs_found) > 10:
            print(f"  ... and {len(revoked_refs_found) - 10} more")

    if DRY_RUN:
        print("\n[patch_revoked_metadata] DRY RUN — no changes written.")
        return

    # Write patched metadata
    backup_path = METADATA_PATH.with_suffix(".json.bak_pre_fix2")
    if not backup_path.exists():
        import shutil
        shutil.copy2(METADATA_PATH, backup_path)
        print(f"[patch_revoked_metadata] Backup saved: {backup_path.name}")

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    print(f"[patch_revoked_metadata] [OK] Patched metadata.json ({total} chunks)")
    print("[patch_revoked_metadata] FAISS pre-filter (vector_store.py) will now skip")
    print("  any chunk where regulatory_status == 'REVOKED' or 'SUPERSEDED'.")
    print("  To revoke a regulation: set chunk['regulatory_status'] = 'REVOKED' and re-run.")


if __name__ == "__main__":
    main()
