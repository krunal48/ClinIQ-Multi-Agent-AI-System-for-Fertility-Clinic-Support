# storage/clinic_db.py
from __future__ import annotations
import sqlite3, time
from pathlib import Path
from typing import List, Dict, Any, Optional

DB = Path("storage/clinic.db")
DB.parent.mkdir(parents=True, exist_ok=True)

def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    _migrate(c)
    return c

def _migrate(c: sqlite3.Connection):
    # appointments
    c.execute("""CREATE TABLE IF NOT EXISTS appointments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        appt_time INTEGER NOT NULL,  -- epoch seconds (UTC)
        tz TEXT DEFAULT 'UTC',
        appt_type TEXT,
        clinician TEXT,
        notes TEXT,
        status TEXT DEFAULT 'scheduled', -- scheduled|completed|cancelled|pending
        ts INTEGER DEFAULT (strftime('%s','now'))
    )""")
    # treatments
    c.execute("""CREATE TABLE IF NOT EXISTS treatments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        regimen TEXT,          -- e.g., IVF antagonist, IUI, FET
        protocol TEXT,         -- meds or protocol name
        start_ts INTEGER,      -- epoch seconds (UTC) nullable
        end_ts INTEGER,        -- epoch seconds (UTC) nullable
        status TEXT DEFAULT 'ongoing', -- ongoing|paused|completed
        notes TEXT,
        ts INTEGER DEFAULT (strftime('%s','now'))
    )""")
    c.commit()

# -------- Appointments --------
def create_appointment(patient_id: str, appt_time_utc: int, tz: str = "UTC",
                       appt_type: str = "", clinician: str = "", notes: str = "",
                       status: str = "scheduled") -> int:
    c = _conn()
    cur = c.execute("""INSERT INTO appointments
        (patient_id, appt_time, tz, appt_type, clinician, notes, status)
        VALUES (?,?,?,?,?,?,?)""",
        (patient_id, appt_time_utc, tz, appt_type, clinician, notes, status))
    c.commit(); appt_id = cur.lastrowid; c.close()
    return appt_id

def list_appointments(patient_id: str, from_utc: Optional[int]=None, to_utc: Optional[int]=None,
                      limit: int = 20) -> List[Dict[str,Any]]:
    q = "SELECT * FROM appointments WHERE patient_id=?"
    params: list[Any] = [patient_id]
    if from_utc is not None: q += " AND appt_time>=?"; params.append(from_utc)
    if to_utc   is not None: q += " AND appt_time<=?"; params.append(to_utc)
    q += " ORDER BY appt_time ASC LIMIT ?"; params.append(limit)
    c = _conn(); rows = [dict(r) for r in c.execute(q, params).fetchall()]; c.close()
    return rows

def cancel_appointment(appt_id: int) -> bool:
    c = _conn(); c.execute("UPDATE appointments SET status='cancelled' WHERE id=?", (appt_id,))
    c.commit(); c.close(); return True

def next_appointment(patient_id: str, now_utc: Optional[int]=None) -> Dict[str,Any] | None:
    now_utc = now_utc or int(time.time())
    c = _conn()
    r = c.execute("""SELECT * FROM appointments
                     WHERE patient_id=? AND appt_time>=? AND status='scheduled'
                     ORDER BY appt_time ASC LIMIT 1""", (patient_id, now_utc)).fetchone()
    c.close()
    return dict(r) if r else None

# -------- Treatments --------
def upsert_treatment(patient_id: str, regimen: str, protocol: str = "",
                     start_ts: Optional[int]=None, end_ts: Optional[int]=None,
                     status: str = "ongoing", notes: str = "") -> int:
    c = _conn()
    # simple rule: if an ongoing record exists, update it; else insert
    r = c.execute("SELECT id FROM treatments WHERE patient_id=? AND status='ongoing' ORDER BY ts DESC LIMIT 1",
                  (patient_id,)).fetchone()
    if r:
        c.execute("""UPDATE treatments
                     SET regimen=?, protocol=?, start_ts=?, end_ts=?, status=?, notes=?, ts=strftime('%s','now')
                     WHERE id=?""",
                  (regimen, protocol, start_ts, end_ts, status, notes, r["id"]))
        tid = r["id"]
    else:
        cur = c.execute("""INSERT INTO treatments
            (patient_id, regimen, protocol, start_ts, end_ts, status, notes)
            VALUES (?,?,?,?,?,?,?)""",
            (patient_id, regimen, protocol, start_ts, end_ts, status, notes))
        tid = cur.lastrowid
    c.commit(); c.close(); return tid

def get_treatment(patient_id: str) -> Dict[str,Any] | None:
    c = _conn()
    r = c.execute("""SELECT * FROM treatments WHERE patient_id=?
                     ORDER BY ts DESC LIMIT 1""", (patient_id,)).fetchone()
    c.close()
    return dict(r) if r else None

def list_treatments(patient_id: str, limit: int = 10) -> List[Dict[str,Any]]:
    c = _conn()
    rows = [dict(r) for r in c.execute("""SELECT * FROM treatments
                   WHERE patient_id=? ORDER BY ts DESC LIMIT ?""", (patient_id, limit)).fetchall()]
    c.close(); return rows
