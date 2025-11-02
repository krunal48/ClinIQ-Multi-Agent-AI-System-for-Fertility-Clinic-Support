
from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
from storage.clinic_db import (
    create_appointment, list_appointments, cancel_appointment, next_appointment
)

def book(patient_id: str, when_utc: int, tz: str = "UTC",
         appt_type: str = "", clinician: str = "", notes: str = "") -> Dict[str,Any]:
    appt_id = create_appointment(patient_id, when_utc, tz, appt_type, clinician, notes, status="scheduled")
    return {"ok": True, "id": appt_id}

def upcoming(patient_id: str, limit: int = 5) -> List[Dict[str,Any]]:
    return list_appointments(patient_id, from_utc=int(time.time()), limit=limit)

def next_one(patient_id: str) -> Dict[str,Any] | None:
    return next_appointment(patient_id)

def cancel(appt_id: int) -> bool:
    return cancel_appointment(appt_id)
