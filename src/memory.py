from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MemoryStore:
    """
    Sliding-window conversational memory.
    Stores (role, message) pairs so the agent can reference prior context.
    """
    _turns: List[Tuple[str, str]] = field(default_factory=list)
    _max_turns: int = 8

    def add_user(self, message: str) -> None:
        self._turns.append(("user", message))
        self._trim()

    def add_assistant(self, message: str) -> None:
        self._turns.append(("assistant", message))
        self._trim()

    def _trim(self) -> None:
        # Keep the last _max_turns * 2 entries (user + assistant pairs)
        self._turns = self._turns[-(self._max_turns * 2):]

    def get_recent_turns(self, n: int = 4) -> List[Tuple[str, str]]:
        """Return the last n (role, message) pairs."""
        return self._turns[-(n * 2):]

    def context_string(self, n: int = 3) -> str:
        """Return a formatted string of the last n conversation turns."""
        turns = self.get_recent_turns(n)
        if not turns:
            return "No prior conversation context."
        lines = []
        for role, msg in turns:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg[:200]}")
        return "\n".join(lines)

    def last_patient_mentioned(self) -> str | None:
        """
        Scan recent user turns for a patient name to support follow-up questions
        like 'what are their medications?' without re-stating the name.
        """
        import re
        known_names = [
            "Rebeca Nagle", "Ramesh Kulkarni", "Anjali Mehra",
            "David Thompson", "Rahul Negi",
        ]
        for _, msg in reversed(self._turns):
            for name in known_names:
                if name.lower() in msg.lower():
                    return name
        # Also catch partial first-name references
        for _, msg in reversed(self._turns):
            match = re.search(r"\bfor ([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b", msg)
            if match:
                return match.group(1)
        return None

    def summary(self) -> str:
        """Short summary for appending to tool responses."""
        turns = self.get_recent_turns(3)
        if not turns:
            return "No recent interactions."
        return " | ".join(f"{r}: {m[:80]}" for r, m in turns)
