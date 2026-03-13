"""One-shot ingestion: CDSCO Draft Guidance on Medical Device Software & SaMD (Oct 2025)."""
import io, sys, requests, pdfplumber
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store

URL = (
    "https://cdsco.gov.in/opencms/resources/UploadCDSCOWeb/2018/"
    "UploadPublic_NoticesFiles/Draft%20guidance%20document%20on%20"
    "Medical%20Device%20Software%2021%2010%202025.pdf"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OrthoLink/1.0)"}

print("Downloading CDSCO SaMD guidance PDF...")
r = requests.get(URL, headers=HEADERS, timeout=60)
r.raise_for_status()
print(f"  {len(r.content):,} bytes")

with pdfplumber.open(io.BytesIO(r.content)) as pdf:
    pages = [p.extract_text() or "" for p in pdf.pages]
text = "\n\n".join(p for p in pages if p.strip())
print(f"  {len(pdf.pages)} pages, {len(text):,} chars")

chunks = chunk_regulatory_text(
    text=text,
    country="IN",
    regulation_name="CDSCO Draft Guidance — Medical Device Software & SaMD (Oct 2025)",
    document_id="IN_CDSCO_SAMD_SOFTWARE_2025",
    source_url=URL,
    device_classes=["Class A", "Class B", "Class C", "Class D", "SaMD"],
)
print(f"  {len(chunks)} chunks created")

store = get_vector_store()
embedded = embed_and_index_chunks(chunks, vector_store=store)
print(f"Done: {embedded} chunks indexed for IN_CDSCO_SAMD_SOFTWARE_2025")
