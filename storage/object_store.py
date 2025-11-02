from __future__ import annotations
import os
import time
import re
from pathlib import Path
from urllib.parse import quote
import requests

FILE_SHARING_BACKEND = os.getenv("FILE_SHARING_BACKEND", "fileio").lower()  # "fileio" | "transfer"
FILEIO_ENDPOINT = os.getenv("FILEIO_ENDPOINT", "https://file.io").rstrip("/")
FILEIO_EXPIRES  = os.getenv("FILEIO_EXPIRES", "14d")
FILEIO_MAX_DL   = os.getenv("FILEIO_MAX_DOWNLOADS")
TRANSFER_ENDPOINT = os.getenv("TRANSFER_ENDPOINT", "https://transfer.sh").rstrip("/")

def _sleep_backoff(attempt: int):
    time.sleep(min(0.5 * (2 ** attempt), 4.0))

def local_file_url(path: str) -> str:
    return f"file://{quote(str(Path(path).resolve()))}"

def _upload_fileio_one(local_path: str, expires: str | None, max_dl: str | None, timeout: int = 30) -> str:
    url = f"{FILEIO_ENDPOINT}/"
    params = {}
    if expires: params["expires"] = expires
    if max_dl:  params["maxDownloads"] = max_dl
    with open(local_path, "rb") as f:
        files = {"file": (Path(local_path).name, f, "application/octet-stream")}
        r = requests.post(url, params=params, files=files, timeout=timeout)
    if r.status_code >= 400:
        msg = r.text.strip()[:300]
        raise RuntimeError(f"file.io HTTP {r.status_code}: {msg or 'no body'}")

    link = None
    ctype = r.headers.get("content-type", "")
    body  = r.text.strip()

    if "application/json" in ctype.lower():
        try:
            data = r.json()
            if not (data.get("success", True) or data.get("status", "").lower() in ("ok","success")):
                raise RuntimeError(data.get("error") or f"file.io error: {data}")
            link = data.get("link") or data.get("url")
        except Exception as e:
            raise RuntimeError(f"file.io JSON parse error: {e}; body starts: {body[:120]}")
    else:
        m = re.search(r"https?://[^\s\"'>]+", body)
        if m:
            link = m.group(0)

    if not link:
        raise RuntimeError(f"file.io response missing link; content-type={ctype}; body starts: {body[:120]}")
    return link

def _upload_fileio(local_paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for lp in local_paths:
        last_err = None
        for attempt in range(0, 3):
            try:
                url = _upload_fileio_one(lp, FILEIO_EXPIRES, FILEIO_MAX_DL)
                out[Path(lp).name] = url
                last_err = None
                break
            except Exception as e:
                last_err = e
                _sleep_backoff(attempt)
        if last_err:
            raise RuntimeError(f"file.io upload failed for {lp}: {last_err}")
    return out

def _upload_transfersh(local_paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for lp in local_paths:
        name = Path(lp).name
        key = f"{int(time.time())}_{name}"
        with open(lp, "rb") as f:
            r = requests.put(f"{TRANSFER_ENDPOINT}/{key}", data=f, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"transfer.sh HTTP {r.status_code}: {r.text[:200]}")
        link = r.text.strip()
        if not re.match(r"^https?://", link):
            raise RuntimeError(f"transfer.sh unexpected body: {link[:120]}")
        out[name] = link
    return out

def upload_and_sign(local_paths: list[str], patient_id: str, case_id: str, expires_s: int = 86400) -> dict[str, str]:
    primary = FILE_SHARING_BACKEND
    try:
        if primary == "fileio":
            return _upload_fileio(local_paths)
        else:
            return _upload_transfersh(local_paths)
    except Exception as e_primary:
        alt = "transfer" if primary == "fileio" else "fileio"
        try:
            if alt == "fileio":
                return _upload_fileio(local_paths)
            else:
                return _upload_transfersh(local_paths)
        except Exception as e_alt:
            raise RuntimeError(f"Sharing failed on both backends ({primary} and {alt}). "
                               f"Primary error: {e_primary}. Fallback error: {e_alt}.")
