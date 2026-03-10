import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

RECORDS_FILE = DATA_DIR / "records.xlsx"
APPOINTMENTS_FILE = DATA_DIR / "appointments.json"
REFERENCE_XML_FILE = DATA_DIR / "medical_reference.xml"
PDF_FILES = [
    DATA_DIR / "sample_patient.pdf",
    DATA_DIR / "sample_report_anjali.pdf",
    DATA_DIR / "sample_report_david.pdf",
    DATA_DIR / "sample_report_ramesh.pdf",
]

# ── LLM configuration ──────────────────────────────────────────────────────
# Set GROQ_API_KEY environment variable to enable LLM-powered synthesis.
# If unset, the assistant runs in retrieval-only mode (no LLM calls).
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Ollama is used as a local fallback if GROQ_API_KEY is not set.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
