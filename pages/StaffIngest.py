from __future__ import annotations

import os, sys
from pathlib import Path
import json
import streamlit as st

# Ensure env vars (.env) are loaded early
import config.env_loader  
sys.path.insert(0, os.getcwd())

from ui.status import env_status_html
from pipelines.ingest_docs import ingest_to_pinecone
from rag.retriever import query_pinecone

# OpenAI (for embedding test queries)
from openai import OpenAI

st.set_page_config(page_title="Staff Ingest & Verify")
st.markdown(env_status_html(), unsafe_allow_html=True)
st.title("Staff — Ingest PDFs & Verify Retrieval")

st.markdown(
    "Upload clinic PDFs (policies, FAQs, education materials), embed, and upsert to **Pinecone**. "
    "Then run a quick retrieval test to confirm the chunks are searchable."
)

with st.sidebar:
    st.header("Settings")
    # Namespace & doc type (ingestion function uses CLINIC_NAMESPACE env; we'll set it before calling)
    namespace = st.text_input("Namespace", value=os.getenv("CLINIC_NAMESPACE", "patient_education"))
    doc_type = st.selectbox("Document type", ["policy", "faq", "guide", "consent", "billing"], index=0)
    st.caption("Namespace is used to segment clinic knowledge (default: patient_education).")

    st.divider()
    st.subheader("Environment")
    st.write(f"**PINECONE_INDEX**: `{os.getenv('PINECONE_INDEX', 'fertility-rag')}`")
    st.write(f"**PINECONE_REGION**: `{os.getenv('PINECONE_REGION', 'us-east-1')}`")
    st.write(f"**PINECONE_API_KEY**: {'✓ set' if os.getenv('PINECONE_API_KEY') else '✕ missing'}")
    st.write(f"**OPENAI_API_KEY**: {'✓ set' if os.getenv('OPENAI_API_KEY') else '✕ missing'}")

st.subheader(":blue[Upload & Ingest]")
pdf = st.file_uploader("PDF to ingest", type=["pdf"])

if st.button("Ingest to Pinecone"):
    if not pdf:
        st.warning("Please select a PDF file.")
    elif not os.getenv("PINECONE_API_KEY"):
        st.error("PINECONE_API_KEY is missing — add it to your .env.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing — add it to your .env.")
    else:
        # Persist file
        tmp = Path("storage/tmp"); tmp.mkdir(parents=True, exist_ok=True)
        pdf_path = tmp / pdf.name
        pdf_path.write_bytes(pdf.getbuffer())

        # Ensure ingest uses the chosen namespace (pipelines/ingest_docs.py reads CLINIC_NAMESPACE)
        os.environ["CLINIC_NAMESPACE"] = namespace

        try:
            res = ingest_to_pinecone(str(pdf_path), doc_type=doc_type)
            st.success(f"Ingested Done → {res}")
            with st.expander("Raw result"):
                st.json(res)
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

st.markdown("---")
st.subheader(":blue[Quick Retrieval Test]")

st.caption(
    "Type a query to verify that your document chunks were indexed. "
    "This searches the **selected namespace** in Pinecone and shows the top matches."
)

query = st.text_input("Test query", value="What is the clinic cancellation policy?")
top_k = st.slider("Top-K", min_value=3, max_value=15, value=8, step=1)

def _embed_query(q: str):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    r = client.embeddings.create(model="text-embedding-3-small", input=[q])
    return r.data[0].embedding

if st.button("Search namespace"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing — cannot embed the query.")
    else:
        try:
            qv = _embed_query(query)
            matches = query_pinecone(qv, k=top_k, namespace=namespace, filters={"pii": {"$eq": False}})
        except Exception as e:
            st.error(f"Query failed: {e}")
            matches = []

        if not matches:
            st.warning("No matches found. If this is a brand-new namespace, try ingesting a PDF first.")
        else:
            st.success(f"Found {len(matches)} matches.")
            # Show a compact list of results
            for i, m in enumerate(matches, 1):
                md = m.get("metadata", {}) if isinstance(m, dict) else {}
                txt = (md.get("text") or "").strip()
                if len(txt) > 400: txt = txt[:400] + "…"
                score = m.get("score", None)
                title = md.get("title") or md.get("source") or md.get("source_file") or "(no title)"
                st.markdown(f"**{i}. {title}**  — score: `{score:.3f}`" if score is not None else f"**{i}. {title}**")
                if txt:
                    st.write(txt)
                meta_show = {k: v for k, v in md.items() if k != "text"}
                if meta_show:
                    with st.expander("metadata"):
                        st.json(meta_show)
                st.markdown("---")

st.info(
    "Note: Our **chat** page will first search the patient namespace (if a Patient ID is set), "
    "then this clinic / different namespace , then fall back to manifest OCR, and finally to OpenAI general guidance."
)
