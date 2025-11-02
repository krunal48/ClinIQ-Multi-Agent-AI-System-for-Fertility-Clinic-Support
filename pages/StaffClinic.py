from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st

import config.env_loader  # load .env

from storage.clinic_db import (
    create_appointment, list_appointments, cancel_appointment,
    upsert_treatment, get_treatment, list_treatments
)

st.set_page_config(page_title="Staff — Appointments & Treatments", page_icon="🗓️")
st.title(" Staff — Appointments & Treatments")

#  Appointments (Create) 
st.subheader("Create appointment")
pid = st.text_input("Patient ID", key="staff_pid")
col1, col2, col3 = st.columns(3)
with col1:
    date_str = st.date_input("Date", key="appt_date").strftime("%Y-%m-%d")
with col2:
    time_str = st.time_input("Time", key="appt_time").strftime("%H:%M")
with col3:
    tz = st.text_input("Time zone (IANA or label)", value="GMT", key="appt_tz")

appt_type = st.text_input("Type (e.g., baseline scan, retrieval consult)", key="appt_type")
clinician = st.text_input("Clinician", key="appt_clinician")
notes = st.text_area("Notes", height=80, key="appt_notes")  # <-- unique key

if st.button("Create appointment", key="btn_create_appt"):
    if not pid:
        st.warning("Enter Patient ID")
    else:
        dt = datetime.fromisoformat(f"{date_str}T{time_str}:00").replace(tzinfo=timezone.utc)
        appt_id = create_appointment(
            pid, int(dt.timestamp()), tz, appt_type, clinician, notes, status="scheduled"
        )
        st.success(f"Appointment #{appt_id} created for {pid}")

st.divider()

#  Appointments (Upcoming / Cancel) 
st.subheader("Upcoming appointments")
pid2 = st.text_input("Patient ID to view", key="view_pid")
if pid2:
    rows = list_appointments(pid2, from_utc=int(time.time()), limit=20)
    if not rows:
        st.info("No upcoming appointments.")
    else:
        for r in rows:
            dt = datetime.fromtimestamp(r["appt_time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            st.write(
                f"- #{r['id']} • {dt} • {r.get('appt_type') or 'visit'} • "
                f"{r.get('clinician') or ''} • status: {r['status']}"
            )
        appt_to_cancel = st.number_input("Cancel appointment ID", min_value=0, step=1, key="cancel_appt_id")
        if st.button("Cancel appointment", key="btn_cancel_appt"):
            if appt_to_cancel > 0:
                cancel_appointment(int(appt_to_cancel))
                st.success(f"Cancelled #{appt_to_cancel}")

st.divider()

#  Treatments 
st.subheader("Treatment plan")
pid3 = st.text_input("Patient ID (treatment)", key="tx_pid")
regimen = st.text_input("Regimen (e.g., IVF antagonist)", key="tx_regimen")
protocol = st.text_input("Protocol/meds (optional)", key="tx_protocol")
t_notes = st.text_area("Notes", height=80, key="tx_notes")  # <-- different key than the appointment notes

if st.button("Set/Update treatment", key="btn_update_tx"):
    if not pid3 or not regimen:
        st.warning("Enter patient + regimen")
    else:
        tid = upsert_treatment(
            pid3, regimen=regimen, protocol=protocol,
            start_ts=int(time.time()), notes=t_notes
        )
        st.success(f"Treatment updated (id={tid})")

if pid3:
    st.caption("Current")
    cur = get_treatment(pid3)
    if cur:
        st.json(cur)
    st.caption("History")
    hist = list_treatments(pid3, limit=10)
    if hist:
        st.json(hist)
