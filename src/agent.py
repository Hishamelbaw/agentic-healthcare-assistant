from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TypedDict

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    END = "__END__"

    class _CompiledGraph:
        def __init__(self, nodes, entry_point, router_fn, edge_map):
            self.nodes = nodes
            self.entry_point = entry_point
            self.router_fn = router_fn
            self.edge_map = edge_map

        def invoke(self, state):
            current = self.entry_point
            while current != END:
                updates = self.nodes[current](state) or {}
                state.update(updates)
                if current == self.entry_point:
                    current = self.edge_map[self.router_fn(state)]
                else:
                    current = END
            return state

    class StateGraph:
        def __init__(self, _state_type):
            self.nodes = {}
            self.entry_point = None
            self.router_fn = None
            self.edge_map = {}

        def add_node(self, name, fn): self.nodes[name] = fn
        def set_entry_point(self, name): self.entry_point = name
        def add_conditional_edges(self, node_name, router_fn, mapping):
            self.router_fn = router_fn
            self.edge_map = mapping
        def add_edge(self, _src, _dst): return None
        def compile(self):
            return _CompiledGraph(self.nodes, self.entry_point, self.router_fn, self.edge_map)

from .appointments import AppointmentManager
from .memory import MemoryStore
from .records import RecordManager
from .retriever import MedicalRetriever

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

from .config import GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL


def _get_llm():
    if GROQ_AVAILABLE and GROQ_API_KEY:
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    if OLLAMA_AVAILABLE:
        try:
            return ChatOllama(model=OLLAMA_MODEL, temperature=0)
        except Exception:
            pass
    return None


class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    response: str


# Intent routing rules: each entry is (intent_key, required_positive_keywords, negative_keywords)
_INTENT_RULES = [
    ("update_record",   ["update patient", "update record", "add note", "add diagnosis", "edit record"], []),
    ("cancel_appointment", ["cancel appointment", "cancel booking", "remove appointment", "delete appointment", "delete"], []),
    ("book_appointment", ["book appointment", "schedule appointment", "make an appointment", "book", "schedule"], ["cancel", "delete", "remove", "show", "list"]),
    ("list_appointments",  ["list appointment", "show appointment", "upcoming appointment", "my appointment", "all appointment"], []),
    ("summarize_history",  ["summarize", "summarise", "medical history", "full history", "history for"], []),
    ("show_record",        ["show patient", "patient record", "show record", "retrieve record",
                             "get record", "find patient", "look up", "medications", "allergies",
                             "contact details", "their record", "his record", "her record"], []),
    ("search_medical_info",["disease", "condition", "medical information", "what is", "what does",
                             "tell me about", "hypertension", "diabetes", "infection", "migraine",
                             "pcos", "costochondritis", "report say", "symptoms", "treatment"], []),
]


@dataclass
class HealthcareAssistant:
    appointment_manager: AppointmentManager = field(default_factory=AppointmentManager)
    record_manager: RecordManager = field(default_factory=RecordManager)
    retriever: MedicalRetriever = field(default_factory=MedicalRetriever)
    memory: MemoryStore = field(default_factory=MemoryStore)

    def __post_init__(self) -> None:
        self._llm = _get_llm()

        graph = StateGraph(AgentState)
        for node_name in [
            "router", "book_appointment", "cancel_appointment", "list_appointments",
            "show_record", "update_record", "summarize_history", "search_medical_info", "fallback",
        ]:
            graph.add_node(node_name, getattr(self, f"_{node_name}"))

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            self._next_node,
            {
                "book_appointment": "book_appointment",
                "cancel_appointment": "cancel_appointment",
                "list_appointments": "list_appointments",
                "show_record": "show_record",
                "update_record": "update_record",
                "summarize_history": "summarize_history",
                "search_medical_info": "search_medical_info",
                "fallback": "fallback",
            },
        )
        for node in [
            "book_appointment", "cancel_appointment", "list_appointments",
            "show_record", "update_record", "summarize_history",
            "search_medical_info", "fallback",
        ]:
            graph.add_edge(node, END)

        self.graph = graph.compile()

    # ── Public entry point ─────────────────────────────────────────────────

    def run(self, user_input: str) -> str:
        self.memory.add_user(user_input)
        # Resolve pronoun / implicit patient references using memory
        resolved_input = self._resolve_patient_reference(user_input)
        state = self.graph.invoke({"user_input": resolved_input})
        response = state.get("response", "I could not process that request.")
        self.memory.add_assistant(response)
        return response

    # ── Pronoun / reference resolution ────────────────────────────────────

    def _resolve_patient_reference(self, text: str) -> str:
        """
        If the user says 'their medications' or 'his record' without naming a patient,
        inject the last mentioned patient name from memory.
        """
        pronouns = ["their", "his", "her", "they", "them", "the patient"]
        text_lower = text.lower()
        needs_resolution = any(p in text_lower for p in pronouns)
        has_name = bool(re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", text))
        if needs_resolution and not has_name:
            last = self.memory.last_patient_mentioned()
            if last:
                return f"{text} (referring to {last})"
        return text

    # ── Intent routing ─────────────────────────────────────────────────────

    def _router(self, state: AgentState) -> AgentState:
        text = state["user_input"].lower()
        for intent, positives, negatives in _INTENT_RULES:
            if any(kw in text for kw in positives) and not any(nk in text for nk in negatives):
                return {"intent": intent}
        return {"intent": "fallback"}

    def _next_node(self, state: AgentState) -> str:
        return state["intent"]

    # ── Extraction helpers ─────────────────────────────────────────────────

    def _extract_patient_name(self, text: str) -> str:
        # First check memory for a recently mentioned patient
        if any(p in text.lower() for p in ["their", "his", "her", "the patient"]):
            last = self.memory.last_patient_mentioned()
            if last:
                return last

        patterns = [
            r"(?:for|of|about|patient record for|history for|update.*?for|summarize.*?for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:'s|\s+record|\s+history|\s+appointment)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        # Last resort: two consecutive title-case words (skip common stop words)
        skip = {"Book", "Show", "Get", "Find", "Update", "List", "Schedule", "Doctor",
                "Patient", "Record", "History", "Appointment", "Medical", "Information"}
        words = re.findall(r"\b[A-Z][a-z]+\b", text)
        name_words = [w for w in words if w not in skip]
        if len(name_words) >= 2:
            return f"{name_words[0]} {name_words[1]}"
        return text.strip()

    def _extract_doctor(self, text: str) -> str:
        match = re.search(r"(Dr\.?\s+[A-Z][a-z]+)", text)
        return match.group(1).replace("Dr.", "Dr.").strip() if match else "Dr. Smith"

    def _extract_date(self, text: str) -> str:
        today = date.today()
        low = text.lower()
        if "tomorrow" in low:
            return (today + timedelta(days=1)).isoformat()
        if "today" in low:
            return today.isoformat()
        # e.g. "2025-04-15"
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if date_match:
            return date_match.group(1)
        return (today + timedelta(days=2)).isoformat()

    def _extract_time(self, text: str) -> str | None:
        explicit = re.search(r"\b(\d{1,2}:\d{2})\b", text)
        if explicit:
            hh, mm = explicit.group(1).split(":")
            return f"{int(hh):02d}:{mm}"
        low = text.lower()
        if "morning" in low:
            return "09:00"
        if "afternoon" in low:
            return "14:00"
        if "evening" in low:
            return "16:00"
        return None

    # ── Tool nodes ─────────────────────────────────────────────────────────

    def _book_appointment(self, state: AgentState) -> AgentState:
        text = state["user_input"]
        patient = self._extract_patient_name(text)
        doctor = self._extract_doctor(text)
        appt_date = self._extract_date(text)
        appt_time = self._extract_time(text)
        response = self.appointment_manager.book(patient, doctor, appt_date, appt_time)
        return {"response": response}

    def _cancel_appointment(self, state: AgentState) -> AgentState:
        text = state["user_input"]
        patient = self._extract_patient_name(text)
        response = self.appointment_manager.cancel(patient)
        return {"response": response}

    def _list_appointments(self, state: AgentState) -> AgentState:
        response = self.appointment_manager.list_all()
        return {"response": response}

    def _show_record(self, state: AgentState) -> AgentState:
        patient = self._extract_patient_name(state["user_input"])
        return {"response": self.record_manager.get_record(patient)}

    def _update_record(self, state: AgentState) -> AgentState:
        text = state["user_input"]
        patient = self._extract_patient_name(text)
        note = text.split(":", 1)[1].strip() if ":" in text else text
        response = self.record_manager.update_record(patient, note)
        return {"response": response}

    def _summarize_history(self, state: AgentState) -> AgentState:
        patient = self._extract_patient_name(state["user_input"])
        raw_summary = self.record_manager.summarize_history(patient)

        # Use LLM to produce a coherent narrative summary from raw data
        if self._llm is not None:
            prompt = (
                "You are a clinical assistant. Given the following raw patient data, "
                "produce a clear, concise medical history summary in 3-5 sentences. "
                "Include key demographics, diagnoses, current treatments, and any notable findings.\n\n"
                f"Raw data:\n{raw_summary}\n\n"
                "Summary:"
            )
            try:
                llm_response = self._llm.invoke(prompt)
                content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
                return {"response": f"Medical History Summary\n{'='*40}\n{content}\n\n--- Raw Data ---\n{raw_summary}"}
            except Exception:
                pass

        return {"response": raw_summary}

    def _search_medical_info(self, state: AgentState) -> AgentState:
        query = state["user_input"]
        result = self.retriever.search(query)
        context = self.memory.context_string(n=2)
        response = result
        if context and context != "No prior conversation context.":
            response += f"\n\n[Conversation context: {context}]"
        return {"response": response}

    def _fallback(self, state: AgentState) -> AgentState:
        return {
            "response": (
                "I can help with the following — please try one of these phrasings:\n\n"
                "  • book an appointment for <Patient Name> with Dr. Smith tomorrow morning\n"
                "  • cancel appointment for <Patient Name>\n"
                "  • list all appointments\n"
                "  • show patient record for <Patient Name>\n"
                "  • update patient record for <Patient Name>: <note>\n"
                "  • summarize medical history for <Patient Name>\n"
                "  • search disease information about <condition>\n"
                "  • what does <Patient Name>'s report say about diabetes?\n\n"
                "Type 'exit' to quit."
            )
        }
