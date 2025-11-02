from __future__ import annotations
import os
from typing import List, Dict, Any
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

def _make_snippets(updates: List[Dict[str, Any]]) -> List[str]:
    out = []
    for u in updates:
        day = u.get("day")
        stage = (u.get("stage") or "").strip()
        total = u.get("total")
        good = u.get("good")
        grades = (u.get("grades") or "").strip()
        notes = (u.get("notes") or "").strip()
        line = f"Day {day} ({stage}): total={total}, good={good}"
        if grades: line += f", grades={grades}"
        if notes:  line += f". Notes: {notes}"
        out.append(line)
    return out

def upsert_updates_to_pinecone(patient_id: str, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not PINECONE_API_KEY:
        return {"mode": "skip", "reason": "PINECONE_API_KEY missing", "count": 0}
    snippets = _make_snippets(updates)
    vectors, dim, backend = embed_texts_robust(snippets)
    if not vectors:
        return {"mode": "skip", "reason": "No vectors", "count": 0, "namespace": f"patient:{patient_id}", "backend": backend}

    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ensure_index(pc, dim)
    index = pc.Index(INDEX_NAME)

    ns = f"patient:{patient_id}"
    upserts = []
    for i, (vec, text) in enumerate(zip(vectors, snippets)):
        upserts.append({
            "id": f"{patient_id}:embryology:{i:06d}",
            "values": vec,
            "metadata": {
                "text": text,            # <-- critical for RAG
                "patient_id": patient_id,
                "pii": True,
                "kind": "embryology_update",
                "day": updates[i].get("day"),
                "stage": updates[i].get("stage"),
            }
        })
    index.upsert(vectors=upserts, namespace=ns)
    return {"mode": "pinecone", "backend": backend, "count": len(upserts), "namespace": ns, "index": INDEX_NAME, "dim": dim}
