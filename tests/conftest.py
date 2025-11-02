# tests/conftest.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple
import pytest
from PIL import Image, ImageDraw

@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Run tests in an isolated temp cwd so registry DB and storage are sandboxed.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)   # for weights path fallback
    (tmp_path / "storage" / "patients").mkdir(parents=True, exist_ok=True)
    (tmp_path / "storage" / "tmp").mkdir(parents=True, exist_ok=True)
    # minimal .env via environment vars for tests
    monkeypatch.setenv("YOLO_WEIGHTS", "models/documents.pt")  # not used because we mock _predict_one
    # leave OPENAI/PINECONE unset; we mock upsert
    yield

@pytest.fixture
def make_dummy_image(tmp_path: Path) -> Tuple[Path, Tuple[int,int,int,int]]:
    """
    Create a simple image with a white rectangle that we pretend YOLO detects.
    Return (image_path, bbox)
    """
    img_path = tmp_path / "dummy.png"
    W, H = 800, 600
    bbox = (200, 220, 600, 300)  # x1,y1,x2,y2
    im = Image.new("RGB", (W, H), (30, 30, 30))
    dr = ImageDraw.Draw(im)
    dr.rectangle(bbox, outline=(250, 250, 250), width=3)
    dr.text((bbox[0]+5, bbox[1]-20), "AMH", fill=(255,255,255))
    im.save(img_path)
    return img_path, bbox
