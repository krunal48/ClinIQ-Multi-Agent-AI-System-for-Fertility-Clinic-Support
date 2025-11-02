# storage/registry.py
from pathlib import Path
import sqlite3
import time
import os

DB = Path("storage/registry.db")
DB.parent.mkdir(parents=True, exist_ok=True)

def _conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    # Ensure table exists
    conn.execute(
        """CREATE TABLE IF NOT EXISTS manifests(
             patient_id   TEXT,
             manifest_path TEXT
             -- ts will be added/migrated below
        )"""
    )
    conn.commit()
    _migrate(conn)
    return conn

def _migrate(conn: sqlite3.Connection):
    """Ensure the schema has a ts INTEGER column; backfill if missing/NULL."""
    # What columns exist now?
    cols = {row[1] for row in conn.execute("PRAGMA table_info(manifests)").fetchall()}
    if "ts" not in cols:
        # Add ts column
        conn.execute("ALTER TABLE manifests ADD COLUMN ts INTEGER")
        conn.commit()
        # Backfill ts for existing rows using file mtime or now()
        rows = conn.execute("SELECT rowid, manifest_path FROM manifests WHERE ts IS NULL").fetchall()
        for r in rows:
            mp = r["manifest_path"]
            try:
                mtime = int(Path(mp).stat().st_mtime)
            except Exception:
                mtime = int(time.time())
            conn.execute("UPDATE manifests SET ts=? WHERE rowid=?", (mtime, r["rowid"]))
        conn.commit()

def register_manifest(patient_id: str, manifest_path: str):
    conn = _conn()
    # prefer file mtime; fallback to now
    try:
        ts = int(Path(manifest_path).stat().st_mtime)
    except Exception:
        ts = int(time.time())
    conn.execute(
        "INSERT INTO manifests (patient_id, manifest_path, ts) VALUES (?, ?, ?)",
        (patient_id, manifest_path, ts),
    )
    conn.commit()
    conn.close()

def latest_manifest(patient_id: str) -> str | None:
    conn = _conn()
    cur = conn.execute(
        "SELECT manifest_path FROM manifests WHERE patient_id=? ORDER BY ts DESC LIMIT 1",
        (patient_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row["manifest_path"] if row else None

def list_manifests(patient_id: str, limit: int = 5):
    conn = _conn()
    cur = conn.execute(
        "SELECT manifest_path, ts FROM manifests WHERE patient_id=? ORDER BY ts DESC LIMIT ?",
        (patient_id, limit),
    )
    rows = [(r["manifest_path"], r["ts"]) for r in cur.fetchall()]
    conn.close()
    return rows
