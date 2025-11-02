import os, json, math
from pathlib import Path
from typing import List, Dict, Any, Optional
import config.env_loader

def _fallback_local(namespace: str, query_vec: List[float], k: int):
    path = Path("storage/pinecone_stub") / f"{namespace}.jsonl"
    if not path.exists(): return []
    vecs = [json.loads(line) for line in path.read_text().splitlines()]
    def cos(a,b):
        s = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
        return 0.0 if na==0 or nb==0 else s/(na*nb)
    scored = sorted([(cos(query_vec, v["values"]), v) for v in vecs], key=lambda x: x[0], reverse=True)
    return [{"score": s, "metadata": v["metadata"]} for s, v in scored[:k]]

def query_pinecone(query_vec: List[float], k: int = 8, namespace: str = "patient_education",
                   filters: Optional[Dict[str, Any]] = None):
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "fertility-rag")
    if not api_key:
        return _fallback_local(namespace, query_vec, k)
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        res = pc.Index(index_name).query(
            vector=query_vec, top_k=k, namespace=namespace,
            include_metadata=True, filter=filters or {}
        )
        return [{"score": m["score"], "metadata": m["metadata"]} for m in res["matches"]]
    except Exception:
        return _fallback_local(namespace, query_vec, k)
