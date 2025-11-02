from __future__ import annotations
import time
from datetime import datetime, timezone
import streamlit as st

import config.env_loader
from storage.embryology_db import add_update, list_updates
from pipelines.embryology_to_pinecone import upsert_updates_to_pinecone

st.set_page_config(page_title="Staff — Embryology Updates", page_icon="🧫")
st.title("Staff — Daily Embryology Updates")

pid = st.text_input("Patient ID", key="emb_pid")
col1, col2, col3 = st.columns(3)
with col1:
    day = st.number_input("Day (0–6)", min_value=0, max_value=10, step=1, value=3, key="emb_day")
with col2:
    stage = st.selectbox("Stage", ["fertilization", "cleavage", "blastocyst", "transfer"], index=2, key="emb_stage")
with col3:
    date = st.date_input("Date (UTC)", key="emb_date")

col4, col5 = st.columns(2)
with col4:
    total = st.number_input("Total embryos observed", min_value=0, step=1, value=6, key="emb_total")
with col5:
    good = st.number_input("Good-quality embryos", min_value=0, step=1, value=3, key="emb_good")

grades = st.text_input("Grades (e.g., 4BB, 3BA, 3BB)", key="emb_grades")
notes  = st.text_area("Notes", height=80, key="emb_notes")

if st.button("Save daily update", key="btn_save_update"):
    if not pid:
        st.warning("Enter Patient ID")
    else:
        dt = datetime.combine(date, datetime.min.time()).replace(tzinfo=timezone.utc)
        uid = add_update(pid, int(day), int(dt.timestamp()), stage, int(total), int(good), grades, notes)
        st.success(f"Saved update #{uid} for {pid}")

        # Upsert all updates (simple strategy) so chat can answer right away
        ups = list_updates(pid)
        info = upsert_updates_to_pinecone(pid, ups)
        if info.get("mode") == "pinecone":
            st.info(f"Pinecone upsert: {info}")
        else:
            st.warning(f"Upsert skipped: {info}")

st.divider()
if pid:
    st.subheader("Current ledger")
    rows = list_updates(pid)
    if not rows:
        st.info("No updates yet.")
    else:
        for r in rows:
            dt = datetime.fromtimestamp(r["date_utc"], tz=timezone.utc).strftime("%Y-%m-%d")
            st.write(f"- Day {r['day']} ({r['stage']}) • total={r['total']} good={r['good']} • grades={r['grades']} • {dt}")
        st.caption("Saved rows are automatically upserted to the patient's Pinecone namespace.")
