# Agentic Healthcare Assistant for Medical Task Automation

A course-project prototype of an **Agentic Healthcare Assistant** that automates four core tasks:

1. **Book medical appointments** — slot discovery, conflict detection, persistent JSON storage
2. **Manage medical records** — retrieve and update structured patient data from `records.xlsx`
3. **Retrieve and summarise patient histories** — cross-references Excel + PDF reports, with optional LLM narrative synthesis
4. **Search medical information** — dual-source RAG: trusted XML reference + FAISS/keyword search over PDF documents, with optional LLM synthesis

---

## Datasets used

| File | Purpose |
|---|---|
| `data/records.xlsx` | Structured patient demographics and summaries |
| `data/sample_patient.pdf` | Rebeca Nagle — 2 clinical encounters with full SOAP notes and labs |
| `data/sample_report_anjali.pdf` | Anjali Mehra — URI visit |
| `data/sample_report_david.pdf` | David Thompson — Type 2 Diabetes follow-up |
| `data/sample_report_ramesh.pdf` | Ramesh Kulkarni — Hypertension checkup |
| `data/medical_reference.xml` | Trusted disease reference parsed with xmltodict/ElementTree fallback: Hypertension, T2DM, URI, Costochondritis, PCOS, Migraine |
| `data/appointments.json` | Persistent appointment store (auto-created) |

---

## Tech stack

| Library | Role |
|---|---|
| `langgraph` | Agentic workflow / state machine routing |
| `pypdf` | PDF text extraction |
| `openpyxl` / `pandas` | Excel record access |
| `faiss-cpu` + `sentence_transformers` | Vector-based RAG retrieval |
| `xmltodict` | XML reference parsing |
| `langchain_groq` | LLM synthesis via Groq API (optional) |
| `langchain_ollama` | Local LLM fallback via Ollama (optional) |

**Graceful degradation:** if FAISS is unavailable the assistant uses keyword retrieval; if no LLM is configured it returns raw retrieved context. The assistant is fully functional without any LLM.

---

## Project structure

```
improved_final/
├── app.py                  # CLI entry point
├── requirements.txt
├── data/
│   ├── records.xlsx
│   ├── sample_patient.pdf
│   ├── sample_report_anjali.pdf
│   ├── sample_report_david.pdf
│   ├── sample_report_ramesh.pdf
│   ├── medical_reference.xml
│   └── appointments.json
├── docs/
│   ├── architecture.md
│   ├── project_overview.md
│   └── uml.puml
└── src/
    ├── config.py       # Paths + LLM config (reads env vars)
    ├── utils.py        # JSON I/O, text normalisation
    ├── memory.py       # Sliding-window conversational memory
    ├── agent.py        # LangGraph workflow + intent routing
    ├── appointments.py # Book / cancel / list appointments
    ├── records.py      # Excel + PDF patient record management
    └── retriever.py    # Dual-source RAG pipeline
```

---

## Installation

```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Enable LLM synthesis (optional but recommended)

```bash
export GROQ_API_KEY=your_key_here       # Linux/Mac
set GROQ_API_KEY=your_key_here          # Windows
```

Get a free key at https://console.groq.com/

---

## Run

```bash
python app.py
```

---

## Example prompts

```
book an appointment for David Thompson with Dr. Smith tomorrow morning
book an appointment for Anjali Mehra with Dr. Patel tomorrow at 09:30
cancel appointment for David Thompson
list all appointments
show patient record for Rahul Negi
update patient record for Rahul Negi: started vitamin D supplements last week
summarize medical history for Rebeca Nagle
search disease information about hypertension
what does David Thompson's report say about diabetes?
tell me about PCOS symptoms
what is the treatment for migraine?
help
exit
```

---

## Architecture overview

```
User Input
    │
    ▼
MemoryStore ──► pronoun/reference resolution
    │
    ▼
LangGraph Router (intent classification)
    │
    ├─► book_appointment  ──► AppointmentManager ──► appointments.json
    ├─► cancel_appointment ──► AppointmentManager
    ├─► list_appointments  ──► AppointmentManager
    ├─► show_record        ──► RecordManager ──► records.xlsx
    ├─► update_record      ──► RecordManager ──► records.xlsx
    ├─► summarize_history  ──► RecordManager + PDFs [+ LLM synthesis]
    ├─► search_medical_info ──► MedicalRetriever
    │       ├─ XML reference lookup
    │       ├─ FAISS/keyword search over PDF chunks
    │       └─ [LLM synthesis of retrieved context]
    └─► fallback
```

---

## Safety note

This assistant returns **administrative and educational information only**. It does not provide diagnosis, emergency advice, or treatment decisions. Not for use in real clinical settings.


## Requirement coverage

- **Agentic workflow:** implemented in `src/agent.py` with `langgraph` and an internal fallback state graph when LangGraph is unavailable.
- **RAG pipeline:** implemented in `src/retriever.py` using XML trusted references plus FAISS/keyword retrieval over the provided PDFs.
- **Memory module:** implemented in `src/memory.py` for follow-up questions and pronoun resolution.
- **Structured data management:** implemented in `src/records.py` using the provided `records.xlsx`.
- **Appointment scheduling:** implemented in `src/appointments.py` with persistent JSON storage.
- **Trusted medical search:** uses the included XML knowledge base and course datasets; optional LLM synthesis is grounded strictly in retrieved context.
