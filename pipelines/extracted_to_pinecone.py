from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Any, List

from pinecone import Pinecone, ServerlessSpec
import config.env_loader
from ml.embedder import embed_texts_robust

INDEX_NAME = os.getenv("PINECONE_INDEX", "fertility-rag")
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

def _collect_texts_from_manifest(manifest: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for pg in manifest.get("pages", []):
        for det in pg.get("detections", []):
            t = det.get("text")
            if t:
                texts.append(t)
    return texts

def upsert_extracted_to_pinecone(manifest_path: str, patient_id: str) -> Dict[str, Any]:
    if not PINECONE_API_KEY:
        return {"mode": "skip", "reason": "PINECONE_API_KEY missing", "count": 0}
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return upsert_manifest(man, patient_id, doc_tag=Path(manifest_path).name)

def upsert_manifest(report: Dict[str, Any], patient_id: str, doc_tag: str | None = None) -> Dict[str, Any]:
    if not PINECONE_API_KEY:
        return {"mode": "skip", "reason": "PINECONE_API_KEY missing", "count": 0}

    ns = f"patient:{patient_id}"
    texts = _collect_texts_from_manifest({"pages": report.get("pages", [])})
    if not texts:
        return {"mode": "skip", "reason": "No OCR text", "count": 0, "namespace": ns}

    # ⬇️ robust embedding (OpenAI → SBERT → hash)
    vectors, dim, backend = embed_texts_robust(texts)
    if not vectors:
        return {"mode": "skip", "reason": "Embedding returned empty", "count": 0, "namespace": ns, "backend": backend}

    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ensure_index(pc, dim)
    index = pc.Index(INDEX_NAME)

    upserts = []
    for i, (vec, chunk) in enumerate(zip(vectors, texts)):
        upserts.append({
            "id": f"{patient_id}:{(doc_tag or 'doc')}:{i:06d}",
            "values": vec,
            "metadata": {
                "text": chunk,
                "patient_id": patient_id,
                "pii": True,
                "kind": "extracted",
                "doc_tag": doc_tag or ""
            }
        })

    index.upsert(vectors=upserts, namespace=ns)
    return {"mode": "pinecone", "count": len(upserts), "namespace": ns, "index": INDEX_NAME, "dim": dim, "backend": backend}
