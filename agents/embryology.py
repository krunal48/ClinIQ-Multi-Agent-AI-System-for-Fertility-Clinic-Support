from pathlib import Path
from typing import Dict, Any, List
from storage.object_store import upload_and_sign, local_file_url
import os

def _collect_paths(manifest: Dict[str, Any], include_crops: bool) -> List[str]:
    paths: List[str] = []
    for p in manifest.get("pages", []):
        img = p.get("annotated_image")
        if img and Path(img).exists():
            paths.append(str(Path(img).resolve()))
        if include_crops:
            for d in p.get("detections", []):
                c = d.get("crop")
                if c and Path(c).exists():
                    paths.append(str(Path(c).resolve()))
    man = manifest.get("manifest")
    if man and Path(man).exists():
        paths.append(str(Path(man).resolve()))
    return paths

def render_summary(manifest: Dict[str, Any], expires_hours: int = 24, include_crops: bool = False) -> Dict[str, Any]:
    patient_id = str(manifest.get("id") or "unknown_patient")
    case_dir = Path(manifest["pages"][0]["annotated_image"]).parent if manifest.get("pages") else Path(".")
    case_id = case_dir.name

    paths = _collect_paths(manifest, include_crops=include_crops)
    backend = os.getenv("FILE_SHARING_BACKEND", "fileio")

    try:
        signed = upload_and_sign(paths, patient_id=patient_id, case_id=case_id, expires_s=expires_hours * 3600)
        manifest_name = Path(manifest.get("manifest", "")).name if manifest.get("manifest") else None
        manifest_url  = signed.get(manifest_name) if manifest_name else None
        page_urls: List[str] = []
        for p in manifest.get("pages", []):
            name = Path(p.get("annotated_image","")).name
            url  = signed.get(name)
            if url:
                page_urls.append(url)
        return {
            "patient_id": patient_id,
            "pages": page_urls,
            "signed_link": manifest_url,
            "expires_in_hours": expires_hours,
            "share_backend": backend,
        }
    except Exception as e:
        page_urls = []
        for p in manifest.get("pages", []):
            ap = p.get("annotated_image")
            if ap and Path(ap).exists():
                page_urls.append(local_file_url(ap))
        manifest_url = None
        man = manifest.get("manifest")
        if man and Path(man).exists():
            manifest_url = local_file_url(man)
        return {
            "patient_id": patient_id,
            "pages": page_urls,
            "signed_link": manifest_url,
            "expires_in_hours": expires_hours,
            "share_warning": f"Sharing unavailable: {e}",
            "share_backend": backend + " (failed → local)",
        }
