from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, List

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "say", "show",
    "tell", "that", "the", "this", "to", "what", "when", "where", "which", "who",
    "with", "would", "about", "please", "can", "could", "do", "does", "did", "info",
    "information", "medical", "disease", "condition", "report", "patient", "history",
}


def normalize_text(value: str) -> str:
    """Lowercase and collapse whitespace for consistent matching."""
    text = re.sub(r"[\r\n\t]+", " ", str(value).strip().lower())
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_text(value: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens, excluding common stopwords."""
    text = normalize_text(value)
    tokens = re.findall(r"[a-z0-9]+", text)
    return [tok for tok in tokens if tok and tok not in STOPWORDS]


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
