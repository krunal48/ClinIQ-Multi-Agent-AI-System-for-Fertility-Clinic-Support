# tests/test_extract_agent.py
from __future__ import annotations
from pathlib import Path
import json
import types
import pytest
from PIL import Image

import agents.extract as ex
import pipelines.document_detector as dd
import storage.registry as reg

def _fake_predict_one(img_pil: Image.Image):
    W, H = img_pil.size
    return [{"label": "FSH", "conf": 0.88, "bbox": [int(W*0.2), int(H*0.25), int(W*0.6), int(H*0.33)]}]

def test_run_extraction_with_upsert(monkeypatch, tmp_path, make_dummy_image):
    img_path, _ = make_dummy_image

    # --- mock YOLO + OCR ---
    class _FakeModel: pass
    monkeypatch.setattr(dd, "_load_yolo", lambda: _FakeModel())
    monkeypatch.setattr(dd, "_predict_one", _fake_predict_one)
    monkeypatch.setattr(dd, "pytesseract", types.SimpleNamespace(
        image_to_string=lambda img, config=None: "7.1 IU/L"
    ))

    # --- mock Pinecone upsert path (agents.extract detects function signature) ---
    # Force agents.extract to use a fake "dict" upsert function
    def _fake_upsert_manifest(report_dict, patient_id: str, doc_tag: str):
        # verify the detector produced expected shape
        assert isinstance(report_dict, dict) and "manifest" in report_dict
        # pretend we inserted N vectors
        total_texts = 0
        for p in report_dict.get("pages", []):
            for d in p.get("detections", []):
                if d.get("text"): total_texts += 1
        return {"mode": "pinecone-mock", "count": total_texts, "namespace": f"patient:{patient_id}"}

    # Patch resolver to return our fake function
    monkeypatch.setattr(ex, "_resolve_upsert", lambda: ("dict", _fake_upsert_manifest))

    # --- run extraction ---
    report = ex.run_extraction(
        str(img_path),
        patient_id="p01",
        enable_ocr=True,
        save_crops=True,
        out_root="storage/patients"
    )

    # --- assertions on report ---
    assert "manifest" in report and Path(report["manifest"]).exists()
    assert "pages" in report and len(report["pages"]) == 1
    dets = report["pages"][0]["detections"]
    assert len(dets) == 1 and dets[0]["label"] == "FSH"
    assert dets[0].get("text") == "7.1 IU/L"
    assert "pinecone_upsert" in report
    assert report["pinecone_upsert"]["mode"] == "pinecone-mock"
    assert report["pinecone_upsert"]["namespace"] == "patient:p01"

    # --- registry should have the manifest recorded ---
    # insert happened via agents.extract.register_manifest
    latest = reg.latest_manifest("p01")
    assert latest and Path(latest).exists()

    # Confirm manifest content is valid JSON
    data = json.loads(Path(latest).read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "pages" in data
