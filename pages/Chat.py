import os, sys, json, importlib
from pathlib import Path
import streamlit as st
import agents.appointments as appts
import agents.treatments as tx
import agents.embryology_results as embupd
from storage.embryology_db import list_updates
import agents.embryology_results as embsum


# Load env early
import config.env_loader  # noqa: F401
sys.path.insert(0, os.getcwd())

# UI status / debug
from ui.status import env_status_html, debug_blob

# Orchestrator
from agents.asha import route

# Agents
import agents.extract as extract
import agents.embryology as embryology
importlib.reload(extract)
importlib.reload(embryology)

# Registry
from storage.registry import latest_manifest, list_manifests

# Hybrid RAG (the version that worked for you)
from rag.qa import answer_hybrid_with_diagnostics

# Page header 
st.set_page_config(page_title="Patient Chat")
st.markdown(env_status_html(), unsafe_allow_html=True)
st.title("Asha Clinic Assistant")
#st.caption("Answers use: patient Pinecone --> clinic KB --> manifest OCR --> OpenAI fallback (always).")

# Sidebar 
with st.sidebar:
    st.header("Patient")
    patient_id = st.text_input(
        "Patient ID (optional for clinic policy; required for personal results)",
        value=st.session_state.get("patient_id", "")
    )

    st.divider()
    st.header(":blue[Upload a report (optional)]")
    upload = st.file_uploader("PDF / image", type=["pdf", "png", "jpg", "jpeg", "tif"])
    if st.button("Process upload now (YOLO + OCR)"):
        if not patient_id:
            st.warning("Enter a Patient ID first.")
        elif not upload:
            st.warning("Choose a file to process.")
        else:
            tmp = Path("storage/tmp"); tmp.mkdir(parents=True, exist_ok=True)
            in_path = tmp / upload.name
            in_path.write_bytes(upload.getbuffer())
            try:
                report = extract.run_extraction(
                    str(in_path),
                    patient_id=patient_id,
                    enable_ocr=True,
                    save_crops=True,
                )
                st.session_state["last_manifest_path"] = report.get("manifest")
                st.success(f"Processed for patient '{patient_id}'.")
                if report.get("pinecone_upsert"):
                    st.info(f"Pinecone upsert: {report['pinecone_upsert']}")
                elif report.get("pinecone_upsert_error"):
                    st.warning(f"Pinecone upsert error: {report['pinecone_upsert_error']}")
            except Exception as e:
                st.error(f"Extraction failed: {e}")

    if patient_id:
        st.divider()
        rows = list_manifests(patient_id, limit=5)
        if rows:
            st.caption("Recent processed reports:")
            for mp, ts in rows:
                p = Path(mp)
                st.write(f"- `{p.parent.name}` -------> `{p.name}`")

    st.divider()
    if st.checkbox("Show debug"):
        st.code(debug_blob(patient_id), language="json")

# Persist patient
if patient_id:
    st.session_state["patient_id"] = patient_id

# Chat history & state 
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "__synthetic_msg__" not in st.session_state:
    st.session_state["__synthetic_msg__"] = None

# render history
for m in st.session_state["chat_history"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Always-on chat input 
typed_msg = st.chat_input("Ask about your results, upload parsing, or clinic info…", key="chat_box")

# Quick actions 
st.markdown("### :blue[**Quick actions**]")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Show my result", width='stretch'):
        st.session_state["__synthetic_msg__"] = "show my embryology result"

with c2:
    if st.button("Clinic policy", width='stretch'):
        st.session_state["__synthetic_msg__"] = "clinic policy information"
with c3:
    if st.button("Ask about my results", width='stretch'):
        # seed a sensible default question that the user can immediately edit next turn
        st.session_state["__synthetic_msg__"] = "ask about my results: how many good embryos on day 5?"

with st.container():
    #st.markdown("**More actions**")
    c4, c5 = st.columns(2)
    with c4:
        if st.button("My next appointment", width='stretch'):
            st.session_state["__synthetic_msg__"] = "when is my next appointment?"
    with c5:
        if st.button("My treatment status", width='stretch'):
            st.session_state["__synthetic_msg__"] = "what treatment am I on?"

# Helpers 
def _load_latest_manifest_for(pid: str):
    mp = st.session_state.get("last_manifest_path") or (latest_manifest(pid) if pid else None)
    if not mp:
        return None
    try:
        return json.loads(Path(mp).read_text(encoding="utf-8"))
    except Exception:
        return None

def _process_pending_upload(pid: str):
    if not upload:
        return "Please attach a PDF/Image in the sidebar first."
    tmp = Path("storage/tmp"); tmp.mkdir(parents=True, exist_ok=True)
    in_path = tmp / upload.name
    in_path.write_bytes(upload.getbuffer())
    try:
        report = extract.run_extraction(str(in_path), patient_id=pid, enable_ocr=True, save_crops=True)
        st.session_state["last_manifest_path"] = report.get("manifest")
        msg = (
            f" Processed and saved for **{pid}**.\n"
            f"- Annotated pages: {len(report.get('pages', []))}\n"
            f"- Manifest: `{report.get('manifest', '')}`"
        )
        if report.get("pinecone_upsert"):
            msg += f"\n- Pinecone upsert: {report['pinecone_upsert']}"
        elif report.get("pinecone_upsert_error"):
            msg += f"\n- Pinecone upsert error: {report['pinecone_upsert_error']}"
        return msg
    except Exception as e:
        return f"Extraction failed: {e}"

# Choose exactly ONE message to process 
message = st.session_state["__synthetic_msg__"] or typed_msg

if message:
    # 1) Show user message
    st.session_state["chat_history"].append({"role": "user", "content": message})

    # 2) Plan with ASHA (but never block on clarify)
    pid = st.session_state.get("patient_id")  # may be None
    plan = route(message, patient_id=pid, has_pending_upload=bool(upload), clinic_namespace="patient_education")
    action = plan.get("action"); params = plan.get("params", {})

    # 3) Execute action—with robust fallback to hybrid answering
    diag = None
    if action == "extract":
        reply = "Please enter your Patient ID in the sidebar first." if not pid else _process_pending_upload(pid)

    elif action == "show_result":
        if not pid:
            reply = "Please enter your Patient ID in the sidebar first."
            diag = None
        else:
            s = embsum.summarize_updates(pid)  # OpenAI narrative (fallback to plain)
            reply = s["markdown"]
            # (optional) preview latest annotated pages
            manifest = _load_latest_manifest_for(pid)
            if manifest:
                imgs = []
                for pg in manifest.get("pages", []):
                    ai = pg.get("annotated_image")
                    if ai and Path(ai).exists():
                        imgs.append(ai)
                if imgs:
                    st.caption("Recent annotated report pages:")
                    for p in imgs[:4]:
                        st.image(p, width='stretch')
            diag = None

    elif action == "answer":
        manifest = _load_latest_manifest_for(pid) if pid else None
        reply, diag = answer_hybrid_with_diagnostics(
            question=message,
            patient_id=pid,
            manifest=manifest,
            clinic_namespace=params.get("namespace", "patient_education"),
        )

    elif action == "clarify":
        # Do NOT stop—still run hybrid RAG --> OpenAI fallback
        manifest = _load_latest_manifest_for(pid) if pid else None
        reply, diag = answer_hybrid_with_diagnostics(
            question=message,
            patient_id=pid,
            manifest=manifest,
            clinic_namespace=params.get("namespace", "patient_education"),
        )
    
    elif action == "appointments":
        if not pid:
            reply = "Please enter your Patient ID to view appointments."
            diag = None
        else:
            nxt = appts.next_one(pid)
            if nxt:
                from datetime import datetime, timezone
                when = datetime.fromtimestamp(nxt["appt_time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                reply = (
                    f"**Your next appointment**\n\n"
                    f"- When: **{when}** ({nxt.get('tz','UTC')})\n"
                    f"- Type: {nxt.get('appt_type') or 'general visit'}\n"
                    f"- Clinician: {nxt.get('clinician') or 'TBD'}\n"
                    f"- Status: {nxt.get('status')}\n\n"
                    "_Tip: If you need to change this, message the clinic or ask a staff member._"
                )
            else:
                reply = "I don’t see any upcoming appointments. Would you like me to request one for you?"
        # No RAG diagnostics for this branch
        diag = None

    elif action == "treatments":
        if not pid:
            reply = "Please enter your Patient ID to view treatment status."
            diag = None
        else:
            cur = tx.status(pid)
            if cur:
                from datetime import datetime, timezone
                started = cur["start_ts"]
                started_str = (datetime.fromtimestamp(started, tz=timezone.utc).strftime("%Y-%m-%d") if started else "N/A")
                reply = (
                    f"**Your treatment plan**\n\n"
                    f"- Regimen: **{cur.get('regimen') or 'N/A'}**\n"
                    f"- Protocol/meds: {cur.get('protocol') or 'N/A'}\n"
                    f"- Status: {cur.get('status')}\n"
                    f"- Started: {started_str}\n"
                    f"- Notes: {cur.get('notes') or '-'}\n\n"
                    "_If anything looks wrong, please contact the care team._"
                )
            else:
                reply = "No treatment plan on file yet. A staff member can add this for you."
        diag = None

    elif action == "results_qa":
        if not pid:
            reply = "Please enter your Patient ID first so I can search your results."
            diag = None
        else:
            # Load patient manifest (optional context) and run hybrid QA
            manifest = _load_latest_manifest_for(pid)
            # Slight nudge in the question so the model leans on patient data
            q = f"(Focus on patient data if available.) {message}"
            reply, diag = answer_hybrid_with_diagnostics(
                question=q,
                patient_id=pid,
                manifest=manifest,
                clinic_namespace=params.get("namespace", "patient_education"),
            )
            # Badge so users see this is a Q&A, not a summary
            reply = "### Result Q&A\n" + reply

    else:
        # Anything unexpected --> answer via hybrid anyway
        manifest = _load_latest_manifest_for(pid) if pid else None
        reply, diag = answer_hybrid_with_diagnostics(
            question=message,
            patient_id=pid,
            manifest=manifest,
            clinic_namespace="patient_education",
        )

    # 4) Render assistant reply + diagnostics
    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
        if diag:
            with st.expander("Grounding used"):
                st.json(diag)

    # 5) Clear synthetic so the chat input stays visible next run
    st.session_state["__synthetic_msg__"] = None
