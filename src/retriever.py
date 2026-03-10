from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import re

try:
    import xmltodict
    XMLTODICT_AVAILABLE = True
except Exception:
    XMLTODICT_AVAILABLE = False

from pypdf import PdfReader

from .config import PDF_FILES, REFERENCE_XML_FILE
from .utils import normalize_text, tokenize_text, unique_preserve_order

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

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


@dataclass
class Chunk:
    source: str
    text: str
    page: int = 0


@dataclass
class ReferenceEntry:
    name: str
    aliases: List[str]
    overview: str
    symptoms: str
    management: str
    source: str
    searchable_phrases: List[str] = field(default_factory=list)
    searchable_tokens: List[str] = field(default_factory=list)


def _get_llm():
    if GROQ_AVAILABLE and GROQ_API_KEY:
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    if OLLAMA_AVAILABLE:
        try:
            return ChatOllama(model=OLLAMA_MODEL, temperature=0)
        except Exception:
            pass
    return None


class MedicalRetriever:
    def __init__(self) -> None:
        self.chunks = self._build_chunks()
        self.references = self._load_reference_entries()
        self.faiss_model = None
        self.index = None
        self._llm = _get_llm()
        if FAISS_AVAILABLE and self.chunks:
            self._build_faiss_index()

    def _read_pdf_pages(self, path: Path) -> List[str]:
        reader = PdfReader(str(path))
        return [(page.extract_text() or "").strip() for page in reader.pages]

    def _chunk_text(self, source: str, text: str, page: int, chunk_size: int = 500) -> List[Chunk]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[Chunk] = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > chunk_size and current:
                chunks.append(Chunk(source=source, text=current.strip(), page=page))
                current = sentence
            else:
                current = (current + " " + sentence).strip()
        if current:
            chunks.append(Chunk(source=source, text=current, page=page))
        return chunks

    def _build_chunks(self) -> List[Chunk]:
        chunks: List[Chunk] = []
        for pdf in PDF_FILES:
            if not pdf.exists():
                continue
            for page_num, text in enumerate(self._read_pdf_pages(pdf), start=1):
                if text:
                    chunks.extend(self._chunk_text(pdf.name, text, page=page_num))
        return chunks

    def _build_search_terms(self, name: str, aliases: str) -> Tuple[List[str], List[str]]:
        phrases = [normalize_text(name)]
        phrases.extend(normalize_text(a) for a in re.split(r"[,;/]", aliases) if a.strip())
        phrases.extend(normalize_text(a) for a in aliases.split() if len(a.strip()) > 2)
        phrases = [p for p in unique_preserve_order(phrases) if p]
        tokens = unique_preserve_order(tokenize_text(" ".join(phrases)))
        return phrases, tokens

    def _load_reference_entries(self) -> List[ReferenceEntry]:
        if not REFERENCE_XML_FILE.exists():
            return []

        if XMLTODICT_AVAILABLE:
            with open(REFERENCE_XML_FILE, "r", encoding="utf-8") as f:
                payload = xmltodict.parse(f.read())
            raw = payload.get("medical_reference", {}).get("disease", [])
            raw_entries = [raw] if isinstance(raw, dict) else list(raw)
        else:
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(REFERENCE_XML_FILE))
            root = tree.getroot()
            raw_entries = [{child.tag: (child.text or "").strip() for child in disease} for disease in root.findall("disease")]

        entries: List[ReferenceEntry] = []
        for item in raw_entries:
            entry = dict(item)
            if "n" in entry and "name" not in entry:
                entry["name"] = entry.pop("n")
            name = str(entry.get("name", "")).strip()
            aliases_text = str(entry.get("aliases", "")).strip()
            phrases, tokens = self._build_search_terms(name, aliases_text)
            entries.append(
                ReferenceEntry(
                    name=name,
                    aliases=[a for a in aliases_text.split() if a],
                    overview=str(entry.get("overview", "")).strip(),
                    symptoms=str(entry.get("symptoms", "")).strip(),
                    management=str(entry.get("management", "")).strip(),
                    source=str(entry.get("source", "N/A")).strip(),
                    searchable_phrases=phrases,
                    searchable_tokens=tokens,
                )
            )
        return entries

    def _build_faiss_index(self) -> None:
        self.faiss_model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = self.faiss_model.encode([chunk.text for chunk in self.chunks], normalize_embeddings=True)
        dim = len(vectors[0])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def _detect_focus(self, query: str) -> str | None:
        q = normalize_text(query)
        if any(word in q for word in ["symptom", "sign", "presentation"]):
            return "symptoms"
        if any(word in q for word in ["treatment", "manage", "management", "therapy", "medication"]):
            return "management"
        if any(word in q for word in ["what is", "overview", "about", "explain"]):
            return "overview"
        return None

    def _reference_score(self, entry: ReferenceEntry, query: str) -> int:
        query_norm = normalize_text(query)
        query_tokens = set(tokenize_text(query))
        score = 0

        for phrase in entry.searchable_phrases:
            if phrase and phrase in query_norm:
                score += 100 if phrase == normalize_text(entry.name) else 80

        token_overlap = query_tokens & set(entry.searchable_tokens)
        score += len(token_overlap) * 10

        if entry.name and normalize_text(entry.name) in query_norm:
            score += 50

        alias_set = {normalize_text(a) for a in entry.aliases}
        if any(alias and alias in query_norm.split() for alias in alias_set):
            score += 60

        return score

    def _reference_lookup(self, query: str, top_k: int = 2) -> List[ReferenceEntry]:
        scored = []
        for entry in self.references:
            score = self._reference_score(entry, query)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def _keyword_search(self, query: str, k: int = 4) -> List[Chunk]:
        q_tokens = set(tokenize_text(query))
        scored = []
        for chunk in self.chunks:
            tokens = set(tokenize_text(chunk.text))
            overlap = q_tokens & tokens
            score = len(overlap)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:k]]

    def _faiss_search(self, query: str, k: int = 4) -> List[Chunk]:
        if self.index is None or self.faiss_model is None:
            return self._keyword_search(query, k=k)
        vector = self.faiss_model.encode([query], normalize_embeddings=True)
        _, idxs = self.index.search(vector, min(k, len(self.chunks)))
        results = [self.chunks[i] for i in idxs[0] if 0 <= i < len(self.chunks)]
        if not results:
            return self._keyword_search(query, k=k)
        return results

    def _format_reference(self, entry: ReferenceEntry, focus: str | None = None) -> str:
        lines = [f"[Trusted Reference: {entry.name} | Source: {entry.source}]"]
        if focus == "symptoms":
            lines.append(f"Symptoms: {entry.symptoms}")
            lines.append(f"Overview: {entry.overview}")
        elif focus == "management":
            lines.append(f"Management: {entry.management}")
            lines.append(f"Overview: {entry.overview}")
        else:
            lines.append(f"Overview: {entry.overview}")
            lines.append(f"Symptoms: {entry.symptoms}")
            lines.append(f"Management: {entry.management}")
        return "\n".join(lines)

    def search(self, query: str) -> str:
        focus = self._detect_focus(query)
        reference_hits = self._reference_lookup(query)
        chunk_hits = self._faiss_search(query) if FAISS_AVAILABLE else self._keyword_search(query)

        parts: List[str] = []
        if reference_hits:
            parts.append(
                "Trusted medical references:\n" + "\n\n".join(self._format_reference(hit, focus) for hit in reference_hits)
            )

        if chunk_hits:
            excerpts = []
            for chunk in chunk_hits[:3]:
                excerpt = re.sub(r"\s+", " ", chunk.text)[:420]
                excerpts.append(f"[Document: {chunk.source} | Page {chunk.page}]\n{excerpt}")
            parts.append("Relevant patient/report excerpts:\n" + "\n\n".join(excerpts))

        if not parts:
            return f"No relevant medical information found for '{query}'."

        raw_context = "\n\n".join(parts)
        if self._llm is not None:
            return self._synthesise(query, raw_context)
        return raw_context

    def _synthesise(self, query: str, context: str) -> str:
        prompt = (
            "You are a healthcare course-project assistant. Using ONLY the context below, "
            "answer the user's question clearly and concisely. "
            "Do not invent facts. Mention the trusted reference if one is present. "
            "If the query asks about symptoms or treatment, prioritise that section. "
            "Close with a brief note that this is informational and not a diagnosis.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        try:
            response = self._llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            return f"{context}\n\n[Note: LLM synthesis unavailable: {exc}]"
