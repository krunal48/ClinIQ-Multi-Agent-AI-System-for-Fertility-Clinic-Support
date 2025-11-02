from __future__ import annotations
from typing import Dict, Any, Optional, List
from storage.clinic_db import upsert_treatment, get_treatment, list_treatments

def set_plan(patient_id: str, regimen: str, protocol: str = "", notes: str = "",
             start_ts: int | None = None) -> Dict[str,Any]:
    tid = upsert_treatment(patient_id, regimen=regimen, protocol=protocol,
                           start_ts=start_ts, status="ongoing", notes=notes)
    return {"ok": True, "treatment_id": tid}

def status(patient_id: str) -> Dict[str,Any] | None:
    return get_treatment(patient_id)

def history(patient_id: str, limit: int = 10) -> List[Dict[str,Any]]:
    return list_treatments(patient_id, limit=limit)
