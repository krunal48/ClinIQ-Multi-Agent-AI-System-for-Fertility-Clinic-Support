from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
import config.env_loader
from ml.embedder import embed_texts_robust

INDEX_NAME = os.getenv("PINECONE_INDEX", "fertility-rag")
NAMESPACE  = os.getenv("CLINIC_NAMESPACE", "patient_education")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION  = os.getenv("PINECONE_REGION", "us-east-1")

def _ensure_index(pc: Pinecone, dim: int):
    names = [x["name"] for x in pc.list_indexes().get("indexes", [])]
    if INDEX_NAME not in names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )

def _split_words(text: str, max_words=800, overlap=120) -> List[str]:
    words = text.split()
    if not words: return []
    out: List[str] = []
    step = max(1, max_words - overlap)
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i+max_words]))
        i += step
    return out

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text("text") or "" for page in doc]
    finally:
        doc.close()
    return "\n".join(pages).strip()

def ingest_to_pinecone(pdf_path: str, doc_type: str = "policy") -> Dict[str, Any]:
    if not PINECONE_API_KEY:
        return {"mode": "skip", "reason": "PINECONE_API_KEY missing", "count": 0}

    p = Path(pdf_path)
    if not p.exists():
        return {"mode": "skip", "reason": f"file not found: {pdf_path}", "count": 0}

    text = extract_text_from_pdf(str(p))
    if not text:
        return {"mode": "skip", "reason": "no extractable text", "count": 0, "namespace": NAMESPACE}

    chunks = _split_words(text)
    if not chunks:
        return {"mode": "skip", "reason": "no chunks after split", "count": 0, "namespace": NAMESPACE}

    # ⬇️ robust embedding (OpenAI → SBERT → hash)
    vectors, dim, backend = embed_texts_robust(chunks)
    if not vectors:
        return {"mode": "skip", "reason": "embedding returned empty", "backend": backend, "count": 0, "namespace": NAMESPACE}

    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ensure_index(pc, dim)
    index = pc.Index(INDEX_NAME)

    upserts = []
    stem = p.stem
    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        upserts.append({
            "id": f"{stem}:{i:06d}",
            "values": vec,
            "metadata": {
                "text": chunk,
                "doc_type": doc_type,
                "title": stem,
                "source": p.name,
                "pii": False,
                "kind": "kb"
            }
        })

    index.upsert(vectors=upserts, namespace=NAMESPACE)
    return {"mode": "pinecone", "count": len(upserts), "namespace": NAMESPACE, "index": INDEX_NAME, "dim": dim, "backend": backend}


#  CLI 
if __name__ == "__main__":
    import argparse, json as _json
    parser = argparse.ArgumentParser(description="Ingest PDF to Pinecone clinic KB.")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--type", default="policy", help="Document type (policy, faq, guide, etc.)")
    args = parser.parse_args()

    result = ingest_to_pinecone(args.pdf_path, doc_type=args.type)
    print(_json.dumps(result, indent=2))
