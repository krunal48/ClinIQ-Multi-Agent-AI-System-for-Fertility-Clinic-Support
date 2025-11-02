import json
import sys, os, importlib
from pathlib import Path
import streamlit as st

# Ensure .env is loaded early (OPENAI / Pinecone)
import config.env_loader  # noqa: F401

# prefer local modules
sys.path.insert(0, os.getcwd())

from ui.status import env_status_html, debug_blob

# Agent + registry
import agents.extract as extract
importlib.reload(extract)
from storage.registry import list_manifests

st.set_page_config(page_title="YOLO Extraction")
st.markdown(env_status_html(), unsafe_allow_html=True)
st.header("Upload lab reports (PDF/images) * Detect fields * Save & Index to Pinecone")

#  Patient & options (sidebar) 
with st.sidebar:
    st.header("Patient")
    patient_id = st.text_input("Patient ID (required)", value=st.session_state.get("patient_id", ""))

    st.divider()
    st.header("Options")
    ocr = st.checkbox("Run OCR on detected fields", value=True,
                      help="Required if you want text answers in chat.")
    crops = st.checkbox("Save individual crops", value=False)

    st.divider()
    st.header("Upload")
    uploaded = st.file_uploader("PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif"])
    process_clicked = st.button("Process upload (YOLO + OCR)")

    st.divider()
    if st.checkbox("Show debug"):
        st.code(debug_blob(patient_id), language="json")

# Persist patient_id in session
if patient_id:
    st.session_state["patient_id"] = patient_id

#  Handle processing 
report = None
if process_clicked:
    if not patient_id:
        st.warning("Please enter a Patient ID in the sidebar first.")
    elif not uploaded:
        st.warning("Please choose a file to process.")
    else:
        tmp = Path("storage/tmp"); tmp.mkdir(parents=True, exist_ok=True)
        in_path = tmp / uploaded.name
        in_path.write_bytes(uploaded.getbuffer())

        try:
            report = extract.run_extraction(
                str(in_path),
                patient_id=patient_id,
                enable_ocr=ocr,
                save_crops=crops
            )
            st.session_state["last_manifest_path"] = report.get("manifest")
            st.success(f"Processed for patient '{patient_id}'.")
        except Exception as e:
            st.error(f"Extraction failed: {e}")

#  Show latest results (current or recent) 
manifest_path = st.session_state.get("last_manifest_path")
if not report and patient_id and not manifest_path:
    rows = list_manifests(patient_id, limit=1)
    if rows:
        manifest_path = rows[0][0]

if manifest_path and Path(manifest_path).exists():
    try:
        report = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Could not read manifest: {e}")

#  Render results (annotated pages, metadata, pinecone status) 
if report:
    st.subheader("Extraction Summary")
    st.write(f"**Source file:** `{report.get('file', '')}`")
    st.write(f"**Manifest:** `{report.get('manifest', '')}`")

    up = report.get("pinecone_upsert")
    up_err = report.get("pinecone_upsert_error")
    if up:
        st.info(f"Pinecone upsert: {up}")
    if up_err:
        st.warning(f"Pinecone upsert error: {up_err}")

    pages = report.get("pages", [])
    if not pages:
        st.warning("No pages in report.")
    else:
        shown = 0
        for page in pages:
            ann = page.get("annotated_image")
            if ann:
                ann_path = Path(ann)
                if not ann_path.is_absolute():
                    ann_path = ann_path.resolve()
                if ann_path.exists():
                    st.markdown(f"**Page {page.get('page','?')}**")
                    st.image(str(ann_path))
                    shown += 1

            dets = page.get("detections", [])
            if dets:
                with st.expander(f"Detections on page {page.get('page','?')} ({len(dets)})"):
                    for i, d in enumerate(dets[:10], 1):
                        label = d.get("label", "FIELD")
                        conf = d.get("conf", None)
                        text = d.get("text", "")
                        crop = d.get("crop", "")
                        line = f"{i}. **{label}**"
                        if conf is not None:
                            line += f" (conf {conf:.2f})"
                        if text:
                            line += f": {text[:200]}{'…' if len(text) > 200 else ''}"
                        if crop and Path(crop).exists():
                            line += f"\n   - crop: `{Path(crop).name}`"
                        st.markdown(line)
            st.markdown("---")

        if shown == 0:
            st.error("No annotated images found on disk. "
                     "Confirm your YOLO weights path in `pipelines/document_detector.py` "
                     "and file permissions.")
else:
    st.info("Enter a Patient ID in the sidebar and upload a file to process.")
