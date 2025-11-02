# storage/embryology_db.py
from __future__ import annotations
import json, sqlite3, time
from pathlib import Path
from typing import Any, Dict, List, Optional

DB = Path("storage/embryology.db")
DB.parent.mkdir(parents=True, exist_ok=True)

def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    _migrate(c)
    return c

def _migrate(c: sqlite3.Connection):
    c.execute("""CREATE TABLE IF NOT EXISTS updates(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        day INTEGER NOT NULL,           -- e.g., 0=OPU, 1..6
        date_utc INTEGER,               -- epoch seconds (UTC)
        stage TEXT,                     -- fertilization / cleavage / blastocyst / transfer
        total INTEGER,                  -- total embryos observed
        good INTEGER,                   -- number meeting criteria (e.g., >= grade threshold)
        grades TEXT,                    -- short free-text like "4BB, 3BA, ..."
        notes TEXT,
        details_json TEXT,              -- JSON blob for per-embryo details if needed
        ts INTEGER DEFAULT (strftime('%s','now'))
    )""")
    c.commit()

def add_update(patient_id: str, day: int, date_utc: int, stage: str,
               total: int, good: int, grades: str, notes: str,
               details: Dict[str, Any] | None = None) -> int:
    details_json = json.dumps(details or {}, ensure_ascii=False)
    c = _conn()
    cur = c.execute("""INSERT INTO updates
        (patient_id, day, date_utc, stage, total, good, grades, notes, details_json)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (patient_id, day, date_utc, stage, total, good, grades, notes, details_json))
    c.commit(); uid = cur.lastrowid; c.close()
    return uid

def list_updates(patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    c = _conn()
    rows = [dict(r) for r in c.execute(
        "SELECT * FROM updates WHERE patient_id=? ORDER BY day ASC, ts ASC LIMIT ?",
        (patient_id, limit)
    ).fetchall()]
    c.close()
    return rows

def latest_update(patient_id: str) -> Dict[str, Any] | None:
    c = _conn()
    r = c.execute(
        "SELECT * FROM updates WHERE patient_id=? ORDER BY day DESC, ts DESC LIMIT 1",
        (patient_id,)
    ).fetchone()
    c.close()
    return dict(r) if r else None
