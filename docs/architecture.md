# Architecture

## High-level Flow

User Input -> Memory Resolution -> LangGraph Router -> Tool Node -> Response

### Tool Nodes

- `book_appointment` -> `AppointmentManager` -> `appointments.json`
- `cancel_appointment` -> `AppointmentManager`
- `list_appointments` -> `AppointmentManager`
- `show_record` -> `RecordManager` -> `records.xlsx`
- `update_record` -> `RecordManager` -> `records.xlsx`
- `summarize_history` -> `RecordManager` + PDFs
- `search_medical_info` -> `MedicalRetriever`
  - trusted XML lookup
  - FAISS retrieval over PDFs
  - optional LLM synthesis

## Retrieval Design

The retrieval pipeline was tightened to improve disease matching quality:

1. Normalize and tokenize the user query
2. Score XML disease entries with exact phrase matches, alias matches, acronym matches, and token overlap
3. Retrieve relevant PDF chunks with FAISS, falling back to keyword search if FAISS is unavailable
4. Return grounded context or synthesize a concise answer with an optional LLM

This prevents loose matches such as returning diabetes when the query is specifically about **PCOS**.
