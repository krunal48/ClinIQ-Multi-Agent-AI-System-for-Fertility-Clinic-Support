from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

import config.env_loader  # ensure .env is loaded before reading env vars
from rag.retriever import query_pinecone

# OpenAI SDK (>=1.0)
try:
    from openai import OpenAI
except Exception as _e:  # defensive import
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = """You are a clinical information assistant for a fertility clinic.

Ground your answers using this strict precedence:
1) Patient Pinecone (patient namespace) — use the 'text' field in metadata.
2) Clinic KB Pinecone (provided namespace) — use the 'text' field in metadata.
3) Patient manifest OCR (if provided).
If none contain what the user needs, provide careful general guidance with OpenAI
and say it may not reflect the patient's documents. Avoid diagnosis; be concise.
If any sources were used, add a short 'Sources' line naming the layers.
"""


#  OpenAI helpers 
def _oai() -> OpenAI:
    """
    Create an OpenAI client using OPENAI_API_KEY.
    Raises RuntimeError with a clear message if the key or SDK is missing.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai>=1.0.0")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing. Add it to your .env.")
    return OpenAI(api_key=key)


def _embed(question: str) -> List[float]:
    """
    Embed the question once using text-embedding-3-small (1536 dims).
    """
    client = _oai()
    r = client.embeddings.create(model="text-embedding-3-small", input=[question])
    return r.data[0].embedding


#  Context collectors 
def _meta_to_text(m: Dict[str, Any]) -> str:
    """
    Defensive extraction of a displayable text from Pinecone metadata.
    Prefers metadata['text'], but tolerates alternative keys used by some pipelines.
    """
    md = m.get("metadata") if "metadata" in m else m
    if not isinstance(md, dict):
        return ""
    for key in ("text", "chunk", "content", "snippet", "body"):
        v = md.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Last resort: compose a short label from common fields
    title = md.get("title") or md.get("source") or ""
    label = md.get("label") or ""
    pieces = [str(x) for x in (title, label) if x]
    return " - ".join(pieces).strip()


def _collect_manifest_text(manifest: Dict[str, Any] | None, cap: int = 8000) -> str:
    """
    Flatten OCR detections from the manifest into a compact text block.
    """
    if not manifest:
        return ""
    parts: List[str] = []
    acc = 0
    for page in manifest.get("pages", []):
        pno = page.get("page", "?")
        for det in page.get("detections", []):
            t = (det.get("text") or "").strip()
            if not t:
                continue
            line = f"[page {pno}] {det.get('label','FIELD')}: {t}"
            parts.append(line)
            acc += len(line)
            if acc > cap:
                return "\n".join(parts)
    return "\n".join(parts)


#  Main entry: hybrid QA with diagnostics 
def answer_hybrid_with_diagnostics(
    question: str,
    patient_id: str | None,
    manifest: Dict[str, Any] | None,
    clinic_namespace: str = "patient_education",
    model: str = "gpt-4o-mini",
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (answer_markdown, diagnostics_dict)

    Strict precedence:
      1) Patient Pinecone (namespace=f"patient:{patient_id}")
      2) Clinic KB Pinecone (namespace=clinic_namespace, pii=False)
      3) Manifest OCR block (from the passed manifest)
      4) If none present → general OpenAI guidance (clearly labeled)

    Diagnostics include counts and which layer(s) were used.
    """
    # 1) Embed the question once
    try:
        qv = _embed(question)
    except Exception as e:
        # If embeddings cannot run, surface the reason clearly.
        return f"OpenAI embeddings error: {e}", {"error": str(e)}

    # 2) Retrieve in the two Pinecone namespaces
    patient_hits: List[Dict[str, Any]] = []
    if patient_id:
        patient_hits = query_pinecone(qv, k=8, namespace=f"patient:{patient_id}", filters={}) or []

    clinic_hits = query_pinecone(
        qv, k=8, namespace=clinic_namespace, filters={"pii": {"$eq": False}}
    ) or []

    manifest_ctx = _collect_manifest_text(manifest)

    # 3) Build grounding in strict order (patient → clinic → manifest)
    used_layers: List[str] = []
    blocks: List[str] = [f"Question:\n{question}"]

    ptexts = [_meta_to_text(m) for m in patient_hits]
    ptexts = [t for t in ptexts if t]
    if ptexts:
        blocks.append("Patient Pinecone text:\n" + "\n".join(ptexts[:10]))
        used_layers.append("Patient Pinecone")

    ktexts = [_meta_to_text(m) for m in clinic_hits]
    ktexts = [t for t in ktexts if t]
    if not ptexts and ktexts:
        blocks.append("Clinic KB text:\n" + "\n".join(ktexts[:8]))
        used_layers.append("Clinic KB")

    if not ptexts and not ktexts and manifest_ctx:
        blocks.append("Patient OCR (manifest):\n" + manifest_ctx)
        used_layers.append("Manifest OCR")

    if not (ptexts or ktexts or manifest_ctx):
        blocks.append(
            "No patient or clinic text was found. Provide careful general guidance."
        )

    # 4) Ask OpenAI (always returns an answer; grounded if we provided blocks above)
    try:
        client = _oai()
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(blocks)},
        ]
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.2)
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI chat error: {e}", {"error": str(e)}

    # 5) Append sources footer (or note general guidance)
    if used_layers:
        answer += "\n\n---\nSources: " + ", ".join(used_layers)
    else:
        answer += "\n\n*Note: general guidance (no patient or clinic context was retrieved).*"

    # 6) Diagnostics block for UI “Grounding used” panel
    diag = {
        "question": question,
        "patient_ns": f"patient:{patient_id}" if patient_id else None,
        "patient_hits": len(ptexts),
        "clinic_ns": clinic_namespace,
        "clinic_hits": len(ktexts),
        "used_layers": used_layers,
        "used_manifest": bool(manifest_ctx) if not (ptexts or ktexts) else False,
        "fallback_general": not (ptexts or ktexts or manifest_ctx),
    }
    return answer, diag
