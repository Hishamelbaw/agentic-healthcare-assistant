# Project Overview

This project implements an **Agentic Healthcare Assistant for Medical Task Automation** aligned to the course-end project brief. The assistant combines a lightweight agentic workflow, retrieval-augmented generation, persistent memory, and structured/unstructured medical data handling.

## Core Features

1. **Book medical appointments**
   - Accepts patient intent in natural language
   - Matches available doctors and appointment slots
   - Persists bookings to `data/appointments.json`

2. **Manage medical records**
   - Reads the provided `data/records.xlsx`
   - Retrieves patient demographics and summaries
   - Appends updated notes to the spreadsheet

3. **Retrieve medical histories**
   - Summarises spreadsheet history and related PDF reports
   - Uses conversational memory to support follow-up questions

4. **Perform medical information searches**
   - Searches a trusted XML disease reference
   - Searches the provided PDF reports with FAISS or keyword fallback
   - Optionally uses an LLM to synthesise grounded answers

## Why this satisfies the project requirements

- **Agentic AI frameworks:** `langgraph` is used for routing requests to specialized tools.
- **RAG pipeline:** the assistant retrieves evidence from XML and PDF sources before answering.
- **Memory module:** recent turns are stored and reused for referential follow-up queries.
- **Trusted medical sources:** the project uses an included medical reference file and course datasets instead of open web search, making the demo reproducible offline.

## Educational scope

This is a **course prototype** for administrative and informational support. It is not intended for diagnosis, emergency triage, or production clinical deployment.
