# ClinIQ: Multi-Agent AI System for Fertility Clinic Support

ClinIQ is a multi-agent AI system designed to enhance patient interaction, automate lab/report processing, and support daily embryology result delivery in fertility clinics. The system improves clarity for patients, reduces staff workload, and creates a smooth and personalized care experience.

---

## 🚀 Key Features

| Agent | Purpose | Capabilities |
|------|---------|--------------|
| **ASHA Conversational Agent (Manager)** | Main patient interface | Handles natural dialogue, detects patient intent, routes tasks to other agents |
| **RAG Medical Information Agent** | Answers clinical/treatment queries | Retrieves context from clinic documents using embeddings + vector search |
| **Embryology Result Agent** | Shares secure embryo progress updates | Generates textual + visual summaries and securely delivers to patients |
| **Digital Data Extraction (DDE) Agent** | Processes lab reports & medical PDFs | OCR + YOLO-based segmentation to extract structured clinical values |

---

## 🧠 System Architecture Overview

- **ASHA Agent** orchestrates conversation flow and agent collaboration.
- **RAG Agent** retrieves accurate medical context from Pinecone vector DB and generates grounded answers.
- **Embryology Agent** uses stored patient‐specific data to produce daily embryo development summaries.
- **Digital Data Extraction Agent** extracts clinical data from uploaded lab/embryology PDFs/images using:
  - YOLO model for region detection
  - Tesseract OCR for text extraction
  - Structured data storage for retrieval & visualization

---

## 🏗️ Tech Stack

| Component | Technology Used |
|---------|----------------|
| UI & Chat Portal | Streamlit |
| Vector Search | Pinecone |
| OCR | Tesseract (pytesseract) |
| PDF/Image Processing | PyMuPDF, OpenCV |
| Embeddings | text-embedding-3-small / SBERT fallback |
| LLM Reasoning | GPT-4o |
| Detection Model | YOLO fine-tuned on clinical report datasets |
| Storage | SQLite + File-based Patient Workspace |

---

## 🎯 Objectives Achieved

- Reduced repetitive clinician explanations of lab results & embryo status.
- Enabled patients to upload reports and receive structured summaries.
- Delivered secure, traceable, and patient-specific embryology result updates.
- Improved clinic efficiency using automation and grounded agent reasoning.

---

## 📌 Future Enhancements

- Integration with hospital EHR systems
- Multi-language patient support
- Telehealth appointment scheduling & calendar sync
- Mobile companion application

---

## 💡 Summary

ClinIQ demonstrates the power of multi-agent AI in healthcare by delivering *clear communication, personalized patient support, and automated clinical workflows* — improving both patient experience and clinic efficiency.
