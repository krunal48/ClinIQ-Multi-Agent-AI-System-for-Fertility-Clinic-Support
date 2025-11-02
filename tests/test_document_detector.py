# tests/test_document_detector.py
from __future__ import annotations
from pathlib import Path
import json
import types
import pytest
from PIL import Image

import pipelines.document_detector as dd

def _fake_predict_one(img_pil: Image.Image):
    # one detection, deterministic
    W, H = img_pil.size
    x1, y1, x2, y2 = int(W*0.25), int(H*0.4), int(W*0.75), int(H*0.5)
    return [{"label": "AMH", "conf": 0.91, "bbox": [x1, y1, x2, y2]}]

@pytest.mark.parametrize("save_crops", [False, True])
def test_detect_documents_image(monkeypatch, tmp_path, make_dummy_image, save_crops):
    img_path, _bbox = make_dummy_image

    # Monkeypatch: bypass YOLO load and prediction
    class _FakeModel: pass
    monkeypatch.setattr(dd, "_load_yolo", lambda: _FakeModel())
    monkeypatch.setattr(dd, "_predict_one", _fake_predict_one)

    # Monkeypatch OCR to avoid needing Tesseract
    monkeypatch.setattr(dd, "pytesseract", types.SimpleNamespace(
        image_to_string=lambda img, config=None: "2.34 ng/mL"
    ))

    out_dir = tmp_path / "storage" / "patients" / "p01"
    rep = dd.detect_documents(
        str(img_path),
        enable_ocr=True,
        save_crops=save_crops,
        out_dir=out_dir
    )

    # Basic report shape
    assert "file" in rep and "manifest" in rep and "pages" in rep
    man = Path(rep["manifest"])
    assert man.exists()

    # Manifest content
    data = json.loads(man.read_text(encoding="utf-8"))
    assert isinstance(data.get("pages"), list) and len(data["pages"]) == 1
    page = data["pages"][0]
    assert page["annotated_image"]
    assert Path(page["annotated_image"]).exists()
    assert len(page["detections"]) == 1
    det = page["detections"][0]
    assert det["label"] == "AMH"
    assert det["conf"] > 0.5
    if save_crops:
        assert "crop" in det and Path(det["crop"]).exists()
    # OCR was mocked
    assert det.get("text") == "2.34 ng/mL"
