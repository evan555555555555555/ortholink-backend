#!/usr/bin/env python3
"""STEP 4 — Verify the vector store has data for UA, US, IN."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.vector_store import get_vector_store

def main():
    store = get_vector_store()
    for country in ["UA", "US", "IN"]:
        count = store.get_chunk_count(country=country)
        results = store.search(
            "medical device registration requirements",
            country=country,
            top_k=5,
        )
        print(f"{country}: {count} chunks in store, {len(results)} from search")
        if results:
            sample = results[0]["text"][:200]
            print(f"  Sample: {sample}...")
    return 0

if __name__ == "__main__":
    sys.exit(main())
