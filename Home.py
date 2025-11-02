import streamlit as st
import base64
from pathlib import Path
import os
os.environ.setdefault("ASHA_APP_PASSWORD", "CliniIQ@ITC")  # or set in your environment / secrets
from auth_gate import check_password
check_password()



st.set_page_config(page_title="Clinic Assistant", page_icon="🤖", layout="wide")

# ---------- Background helper ----------
def set_background(image_path: str):
    p = Path(image_path)
    if not p.exists():
        st.warning(f"Background image not found: {image_path}")
        return
    b64 = base64.b64encode(p.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Glass card for main container */
        .block-container {{
            background: rgba(255,255,255,0.78);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border-radius: 18px;
            padding: 2.2rem 2.4rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.12);
        }}

        /* Header transparent */
        [data-testid="stHeader"] {{ background: transparent; }}

        /* Gradient headline */
        .hero-title {{
            font-size: clamp(2rem, 3.2vw, 3.4rem);
            font-weight: 800;
            line-height: 1.05;
            letter-spacing: -0.02em;
            background: linear-gradient(92deg, #0ea5e9 0%, #22c55e 50%, #f59e0b 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 1px 0 rgba(255,255,255,0.35);
            margin: 0 0 .25rem 0;
        }}

        /* Animated subtitle */
        @keyframes fadeUp {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .hero-sub {{
            font-size: clamp(1rem, 1.2vw, 1.25rem);
            color: #0f172a;
            opacity: 0.95;
            animation: fadeUp .6s ease both .05s;
            margin-bottom: 1.2rem;
        }}

        /* Feature cards */
        .card {{
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(15,23,42,0.06);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 6px 24px rgba(0,0,0,0.08);
            height: 100%;
        }}
        .card h4 {{
            margin: 0 0 .25rem 0;
            font-size: 1.05rem;
        }}
        .chip {{
            display: inline-block;
            padding: .25rem .55rem;
            border-radius: 999px;
            background: rgba(14,165,233,0.12);
            border: 1px solid rgba(14,165,233,0.25);
            font-size: .78rem;
            margin-right: .35rem;
        }}

        /* Prettier page_link buttons */
        a[data-testid="stPageLink"] div[role="button"] {{
            border-radius: 999px;
            padding: .6rem 1rem;
            font-weight: 700;
            box-shadow: 0 8px 22px rgba(2,132,199,0.22);
        }}

        /* Small helper text */
        .muted {{
            color:#334155; opacity:.9; font-size:.95rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image (uploaded to /mnt/data/ai-chatbot.jpg)
#set_background("C:\\Users\\User\\OneDrive\\Documents\\Agentic_AI_Project\\ai-chatbot.jpg")
set_background("C:\\Users\\kruna\\training\\Projects\\Project 3\\AshaAgent\\conversational-ai.png")
#st.title("Asha Fertility Clinic Assistant")

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero-title">Asha Fertility Clinic Assistant</div>
    <div class="hero-sub">Smart, secure, and explainable—built for patient questions, staff workflows, and daily embryology updates.</div>
    """,
    unsafe_allow_html=True
)

# ---------- Quick actions ----------
colA, colB, colC = st.columns([1,1,1])

with colA:
    st.page_link("pages/Extract.py", label="Extract from Documents", help="OCR + parsing + structuring")
with colB:
    st.page_link("pages/StaffIngest.py", label="Staff — Ingest PDFs & Verify Retrieval", help="Bulk load clinic docs")
with colC:
    st.page_link("pages/StaffClinic.py", label=" Appointments & Treatments", help="Navigate staff tools")

st.markdown("")

# ---------- Feature cards ----------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="card">
          <h4>Precision Answers</h4>
          <p class="muted">Hybrid RAG + OpenAI for context-aware responses.</p>
          <span class="chip">RAG</span><span class="chip">Context Memory</span>
        </div>
        """, unsafe_allow_html=True
    )
with c2:
    st.markdown(
        """
        <div class="card">
          <h4>Embryology Updates</h4>
          <p class="muted">Auto-generated daily summaries with links &amp; visual cues for patients.</p>
          <span class="chip">Secure Links</span>
        </div>
        """, unsafe_allow_html=True
    )
with c3:
    st.markdown(
        """
        <div class="card">
          <h4>Document Intelligence</h4>
          <p class="muted">YOLO + OCR pipeline turns PDFs &amp; images into structured facts.</p>
          <span class="chip">YOLO</span><span class="chip">OCR</span><span class="chip">Pinecone</span>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("")
st.caption("Tip: Use the sidebar to switch pages anytime. Thank you !")

#st.write("Use the sidebar (Pages) or the links below:")

# These work now that pages are registered under ./pages
#st.page_link("pages/Extract.py", label="Go to Extraction Page →")
#st.page_link("pages/StaffIngest.py", label="Go to Staff Ingestion →")
