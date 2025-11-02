from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional dependencies (handled defensively)
try:
    import fitz  # PyMuPDF for PDF raster
except Exception:
    fitz = None  # type: ignore

try:
    import pytesseract  # OCR (optional)
except Exception:
    pytesseract = None  # type: ignore

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None  # type: ignore

# ---------------- Config ----------------
# You can override via .env
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "models/documents.pt")
PDF_DPI = int(os.getenv("PDF_DPI", "220"))
LINE_WIDTH = int(os.getenv("BOX_THICKNESS", "3"))

# --------------- Utilities ---------------
def _now_run_id() -> str:
    return time.strftime("run-%Y%m%d-%H%M%S")

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_yolo():
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics not installed. Run `pip install ultralytics` "
            "and ensure your weights file exists."
        )
    weights = Path(YOLO_WEIGHTS)
    if not weights.exists():
        # Also allow a project-relative common location
        alt1 = Path("documents.pt")
        alt2 = Path("weights/documents.pt")
        for alt in (alt1, alt2):
            if alt.exists():
                return YOLO(str(alt))
        raise FileNotFoundError(
            f"YOLO weights not found at {weights}. "
            "Set YOLO_WEIGHTS in .env to your 'documents.pt' path."
        )
    return YOLO(str(weights))

def _rasterize_pdf(pdf_path: Path, out_dir: Path, dpi: int = PDF_DPI) -> List[Path]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF input. `pip install pymupdf`")
    doc = fitz.open(str(pdf_path))
    paths: List[Path] = []
    try:
        for i, page in enumerate(doc):
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = out_dir / f"page_{i+1:02d}.png"
            pix.save(str(img_path))
            paths.append(img_path)
    finally:
        doc.close()
    return paths

def _load_image_any(path: Path) -> Image.Image:
    im = Image.open(str(path)).convert("RGB")
    return im

def _ocr_pil(img_pil: Image.Image) -> str:
    if pytesseract is None:
        return ""  # OCR gracefully skipped
    try:
        # You can tweak as needed; psm 6 is good for blocks
        return pytesseract.image_to_string(img_pil, config="--psm 6").strip()
    except Exception:
        return ""

def _draw_box(draw: ImageDraw.ImageDraw, box: Tuple[int,int,int,int], color=(0, 255, 0), lw: int = LINE_WIDTH):
    x1, y1, x2, y2 = box
    for k in range(lw):
        draw.rectangle([x1-k, y1-k, x2+k, y2+k], outline=color)

def _save_crop(img: Image.Image, box: Tuple[int,int,int,int], out_dir: Path, label: str, idx: int) -> Path:
    x1, y1, x2, y2 = box
    crop = img.crop((x1, y1, x2, y2))
    crop_path = out_dir / f"crop_{label}_{idx:03d}.png"
    crop.save(str(crop_path))
    return crop_path

def _annotate_page(img: Image.Image, dets: List[Dict[str, Any]]) -> Image.Image:
    # Draw boxes + labels
    im = img.copy()
    draw = ImageDraw.Draw(im)
    for d in dets:
        box = tuple(map(int, d["bbox"]))
        _draw_box(draw, box)
        # label text box
        label = f"{d.get('label','FIELD')} {d.get('conf', 0):.2f}"
        # simple text (PIL default font)
        tx, ty = box[0] + 4, box[1] + 4
        draw.text((tx, ty), label, fill=(255, 0, 0))
    return im

# --------------- Inference core ---------------
def _predict_one(model, img_pil: Image.Image) -> List[Dict[str, Any]]:
    """
    Run YOLO on a PIL image and return a list of detections:
      [{"label": str, "conf": float, "bbox": [x1,y1,x2,y2]}]
    """
    # Ultralytics models accept numpy arrays or PIL
    res = model.predict(img_pil, verbose=False)
    dets: List[Dict[str, Any]] = []
    # Some versions return a list; iterate robustly
    for r in res:
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", None) or getattr(model, "names", {})
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.zeros((xyxy.shape[0],), dtype=np.float32)
        cls  = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.zeros((xyxy.shape[0],), dtype=int)
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i].tolist()
            c  = float(conf[i])
            k  = int(cls[i])
            lbl = str(names.get(k, f"class_{k}"))
            dets.append({"label": lbl, "conf": c, "bbox": [int(x1), int(y1), int(x2), int(y2)]})
    return dets

def _run_page(
    model,
    img_path: Path,
    crops_dir: Path,
    enable_ocr: bool,
    save_crops: bool
) -> Tuple[Dict[str, Any], Path]:
    """
    Process a single raster image page. Returns:
      page_entry (dict for manifest), annotated_path
    """
    img = _load_image_any(img_path)
    dets = _predict_one(model, img)

    page_entry: Dict[str, Any] = {
        "page": int("".join([c for c in img_path.stem if c.isdigit()]) or "1"),
        "raster_image": str(img_path.resolve()),
        "annotated_image": "",
        "detections": [],
    }

    # Per detection: crop (+OCR), fill manifest
    for i, d in enumerate(dets, 1):
        entry = {
            "label": d["label"],
            "conf": float(d["conf"]),
            "bbox": d["bbox"],
        }
        crop_path = None
        if save_crops:
            crop_path = _save_crop(img, tuple(d["bbox"]), crops_dir, d["label"], i)
            entry["crop"] = str(crop_path.resolve())
        if enable_ocr:
            # OCR on crop if we have it, else on full image box
            if crop_path:
                txt = _ocr_pil(Image.open(str(crop_path)).convert("RGB"))
            else:
                x1, y1, x2, y2 = d["bbox"]
                txt = _ocr_pil(img.crop((x1, y1, x2, y2)))
            if txt:
                entry["text"] = txt
        page_entry["detections"].append(entry)

    # Annotated page
    ann = _annotate_page(img, dets)
    ann_path = img_path.parent / f"{img_path.stem}_annotated.png"
    ann.save(str(ann_path))
    page_entry["annotated_image"] = str(ann_path.resolve())

    return page_entry, ann_path

# --------------- Public API ---------------
def detect_documents(
    file_path: str,
    enable_ocr: bool = True,
    save_crops: bool = True,
    out_dir: Union[str, Path] = "storage/patients/_unspecified"
) -> Dict[str, Any]:
    """
    Detect document fields on a PDF or image and produce a manifest JSON.

    Returns a report dict:
      {
        "file": "<input path>",
        "manifest": "<path/to/manifest.json>",
        "pages": [
          {
            "page": 1,
            "raster_image": ".../page_01.png",
            "annotated_image": ".../page_01_annotated.png",
            "detections": [
              {"label": "AMH", "conf": 0.89, "bbox":[x1,y1,x2,y2], "crop":".../crop_AMH_001.png", "text":"..."}
            ]
          },
          ...
        ]
      }
    """
    file_path = str(file_path)
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    # Create run folder
    out_dir = _ensure_dir(out_dir)
    run_dir = _ensure_dir(out_dir / _now_run_id())
    pages_dir = _ensure_dir(run_dir / "pages")
    crops_dir = _ensure_dir(run_dir / "crops")

    # 1) Rasterize or copy image
    raster_paths: List[Path] = []
    if src.suffix.lower() == ".pdf":
        raster_paths = _rasterize_pdf(src, pages_dir, dpi=PDF_DPI)
    else:
        # Treat as single image
        dst = pages_dir / f"page_01{src.suffix.lower()}"
        Image.open(str(src)).convert("RGB").save(str(dst))
        raster_paths = [dst]

    # 2) Load YOLO
    model = _load_yolo()

    # 3) Process pages
    pages: List[Dict[str, Any]] = []
    for rp in raster_paths:
        entry, _ = _run_page(model, rp, crops_dir, enable_ocr=enable_ocr, save_crops=save_crops)
        pages.append(entry)

    # 4) Build manifest + write
    manifest = {
        "file": str(Path(file_path).resolve()),
        "pages": pages,
    }
    man_path = run_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "file": str(Path(file_path).resolve()),
        "manifest": str(man_path.resolve()),
        "pages": pages,
    }
    return report
