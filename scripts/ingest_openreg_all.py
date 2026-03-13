"""Ingest all 91 OpenRegulatory templates not yet in FAISS."""
import sys, json, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store

with open("/tmp/openreg_files.json") as f:
    files = json.load(f)

BASE = "https://raw.githubusercontent.com/openregulatory/templates/master/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
store = get_vector_store()
existing = set(m.get("document_id", "") for m in store.metadata)
print(f"Vector store loaded: {len(store.metadata)} chunks, {len(existing)} unique docs")
total = 0
skipped = 0

for path in files:
    name = (
        path.replace("templates/", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".md", "")
        .upper()
    )
    doc_id = f"EU_OPENREG_{name}"

    if doc_id in existing:
        skipped += 1
        continue

    try:
        r = requests.get(BASE + path, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  SKIP {doc_id}: {e}")
        continue

    text = r.text
    if len(text.split()) < 30:
        print(f"  SKIP {doc_id}: too short")
        continue

    country = "STANDARDS" if any(x in path for x in ("information_security", "data_protection")) else "EU"
    title = path.split("/")[-1].replace(".md", "").replace("-", " ").title()

    chunks = chunk_regulatory_text(
        text=text,
        country=country,
        regulation_name=f"OpenRegulatory Template — {title}",
        document_id=doc_id,
        source_url=f"https://github.com/openregulatory/templates/blob/master/{path}",
        device_classes=["Class I", "Class IIa", "Class IIb", "Class III", "SaMD", "IVD"],
    )
    embedded = embed_and_index_chunks(chunks, vector_store=store)
    total += embedded
    print(f"  +{embedded:3d}  {doc_id}", flush=True)

print(f"\nDone: {skipped} already indexed, {total} new chunks added")
