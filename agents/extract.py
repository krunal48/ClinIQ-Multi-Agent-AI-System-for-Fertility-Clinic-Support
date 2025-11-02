from __future__ import annotations

from typing import Dict, Any
from pathlib import Path

from pipelines.document_detector import detect_documents
from storage.registry import register_manifest

# Support either naming style in your project:
# - upsert_manifest(report_dict, patient_id, doc_tag)
# - upsert_extracted_to_pinecone(manifest_path, patient_id)
_UPSERT_MODE = None
def _resolve_upsert():
    global _UPSERT_MODE
    if _UPSERT_MODE is not None:
        return _UPSERT_MODE
    try:
        # Newer helper taking whole manifest dict
        from pipelines.extracted_to_pinecone import upsert_manifest as _u1  # type: ignore
        _UPSERT_MODE = ("dict", _u1)
        return _UPSERT_MODE
    except Exception:
        pass
    try:
        # Alternative helper taking manifest file path
        from pipelines.extracted_to_pinecone import upsert_extracted_to_pinecone as _u2  # type: ignore
        _UPSERT_MODE = ("path", _u2)
        return _UPSERT_MODE
    except Exception:
        pass
    _UPSERT_MODE = ("none", None)
    return _UPSERT_MODE


def run_extraction(
    file_path: str,
    patient_id: str | None = None,
    enable_ocr: bool = True,
    save_crops: bool = True,
    out_root: str | Path = "storage/patients",
    **kwargs,
) -> Dict[str, Any]:
    """
    Run YOLO(+OCR), save under storage/patients/<patient_id>/..., register manifest,
    and upsert OCR text to Pinecone namespace patient:<patient_id>.

    Returns the 'report' dict from document_detector, plus any Pinecone upsert info.
    """
    if not patient_id:
        patient_id = "unknown_patient"

    out_dir = Path(out_root) / str(patient_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Detect + annotate (+OCR)
    report = detect_documents(
        file_path,
        enable_ocr=enable_ocr,
        save_crops=save_crops,
        out_dir=out_dir,
    )

    # 2) Register manifest path in local registry (for "latest" lookup)
    try:
        register_manifest(str(patient_id), report["manifest"])
    except Exception as e:
        report["registry_error"] = str(e)

    # 3) Upsert to Pinecone (either helper name)
    mode, fn = _resolve_upsert()
    try:
        if mode == "dict" and fn:
            up_res = fn(report, patient_id=str(patient_id), doc_tag=Path(file_path).name)
        elif mode == "path" and fn:
            up_res = fn(report["manifest"], patient_id=str(patient_id))
        else:
            up_res = {"mode": "skip", "reason": "No upsert function found in pipelines.extracted_to_pinecone"}
        report["pinecone_upsert"] = up_res
    except Exception as e:
        report["pinecone_upsert_error"] = str(e)

    return report
