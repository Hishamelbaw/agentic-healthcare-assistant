"""
Agentic Healthcare Assistant — CLI entry point
Run: python app.py
Set GROQ_API_KEY environment variable to enable LLM-powered synthesis.
"""
import os
import sys

from src.agent import HealthcareAssistant
from src.config import GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       Agentic Healthcare Assistant                       ║
║       Medical Task Automation — Course Project           ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Available commands:
  book an appointment for <Name> with Dr. <X> tomorrow morning
  cancel appointment for <Name>
  list all appointments
  show patient record for <Name>
  update patient record for <Name>: <note>
  summarize medical history for <Name>
  search disease information about <condition>
  what does <Name>'s report say about <topic>?
  help        — show this message
  exit/quit   — exit the assistant

Known patients: Rebeca Nagle, Ramesh Kulkarni, Anjali Mehra, David Thompson, Rahul Negi
Available doctors: Dr. Smith, Dr. Patel, Dr. Lee, Dr. Sharma, Dr. Chen
"""


def _print_llm_status() -> None:
    if GROQ_API_KEY:
        print(f"[LLM] Groq enabled — model: {GROQ_MODEL}")
    else:
        try:
            from langchain_ollama import ChatOllama  # noqa: F401
            print(f"[LLM] Ollama fallback enabled — model: {OLLAMA_MODEL}")
        except Exception:
            print("[LLM] No LLM configured. Running in retrieval-only mode.")
            print("      Set GROQ_API_KEY to enable LLM-powered synthesis.")


def main() -> None:
    print(BANNER)
    _print_llm_status()
    print("\nType 'help' for available commands. Type 'exit' to quit.\n")

    try:
        assistant = HealthcareAssistant()
    except Exception as exc:
        print(f"[ERROR] Failed to initialise assistant: {exc}")
        sys.exit(1)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Assistant: Goodbye.")
            break
        if user_input.lower() in {"help", "?"}:
            print(HELP_TEXT)
            continue

        result = assistant.run(user_input)
        print(f"\nAssistant:\n{result}\n")


if __name__ == "__main__":
    main()
