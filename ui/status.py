# ui/status.py
import os

def _badge(ok: bool, label: str) -> str:
    return f"""<span style="
        display:inline-block;padding:3px 8px;border-radius:999px;
        font-size:12px;font-weight:600;line-height:1;
        color:{'#0b3d02' if ok else '#5c0000'};
        background:{'#c7f7c5' if ok else '#ffd6d6'};
        border:1px solid { '#89d487' if ok else '#ff9c9c' };
        margin-right:6px;margin-bottom:6px;
    ">{'✓' if ok else '✕'} {label}</span>"""

def env_status_html() -> str:
    """Return small HTML badges for top-of-page status (no network calls)."""
    # Keys & packages
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    try:
        import openai  # noqa: F401
        openai_ok = True
    except Exception:
        openai_ok = False

    has_pc_key   = bool(os.getenv("PINECONE_API_KEY"))
    has_pc_index = bool(os.getenv("PINECONE_INDEX"))
    has_pc_region = True if os.getenv("PINECONE_REGION") or True else False  # default OK
    try:
        import pinecone  # noqa: F401
        pinecone_ok = True
    except Exception:
        pinecone_ok = False

    # Sharing backend: prefer S3 if bucket set, otherwise file.io
    has_s3_bucket = bool(os.getenv("S3_BUCKET"))
    fileio_endpoint = os.getenv("FILEIO_ENDPOINT", "https://file.io")
    share_backend = "S3" if has_s3_bucket else "file.io"

    parts = [
        _badge(has_openai_key, "OpenAI key"),
        _badge(openai_ok, "openai pkg"),
        _badge(has_pc_key, "Pinecone key"),
        _badge(has_pc_index, "Pinecone index"),
        _badge(has_pc_region, "region"),
        _badge(pinecone_ok, "pinecone pkg"),
        _badge(has_s3_bucket or True, f"share: {share_backend}"),
        _badge(True, f"file.io: {fileio_endpoint.replace('https://','')}"),
    ]
    return "".join(parts)

def debug_blob(patient_id: str | None = None) -> dict:
    """Structured env snapshot for sidebar 'Show debug' panel."""
    blob = {
        # OpenAI
        "OPENAI_KEY": bool(os.getenv("OPENAI_API_KEY")),
        # Pinecone
        "PINECONE_KEY": bool(os.getenv("PINECONE_API_KEY")),
        "PINECONE_INDEX": os.getenv("PINECONE_INDEX"),
        "PINECONE_REGION": os.getenv("PINECONE_REGION", "(default)"),
        # Sharing backends
        "S3_BUCKET": os.getenv("S3_BUCKET"),             # None means using file.io path
        "S3_PREFIX": os.getenv("S3_PREFIX"),
        "FILEIO_ENDPOINT": os.getenv("FILEIO_ENDPOINT", "https://file.io"),
        "FILEIO_EXPIRES": os.getenv("FILEIO_EXPIRES", "14d"),
        "FILEIO_MAX_DOWNLOADS": os.getenv("FILEIO_MAX_DOWNLOADS"),
    }
    if patient_id:
        blob["patient_ns"] = f"patient:{patient_id}"
        blob["clinic_ns"] = "patient_education"
    return blob
