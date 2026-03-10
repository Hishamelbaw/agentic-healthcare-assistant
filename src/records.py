from __future__ import annotations

import math
import re
from typing import Dict, List

import pandas as pd
from pypdf import PdfReader

from .config import PDF_FILES, RECORDS_FILE
from .utils import normalize_text, unique_preserve_order


class RecordManager:
    def __init__(self) -> None:
        self.df = pd.read_excel(RECORDS_FILE)
        self.pdf_texts = self._load_pdf_texts()

    def _load_pdf_texts(self) -> Dict[str, str]:
        texts: Dict[str, str] = {}
        for path in PDF_FILES:
            reader = PdfReader(str(path))
            extracted = []
            for page in reader.pages:
                extracted.append(page.extract_text() or "")
            texts[path.name] = "\n".join(extracted)
        return texts

    def _reload(self) -> None:
        self.df = pd.read_excel(RECORDS_FILE)

    def _clean_value(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip()

    def _match_rows(self, query: str) -> pd.DataFrame:
        needle = normalize_text(query)
        for column in ["Name", "Phone_number", "Email"]:
            mask = self.df[column].astype(str).str.lower().str.contains(needle, na=False)
            if mask.any():
                return self.df[mask]
        return self.df.iloc[0:0]

    def _merge_summaries(self, rows: pd.DataFrame) -> str:
        summaries = [self._clean_value(v) for v in rows["Summary"].tolist() if self._clean_value(v)]
        merged = unique_preserve_order(summaries)
        return " ".join(merged).strip()

    def get_record(self, query: str) -> str:
        self._reload()
        rows = self._match_rows(query)
        if rows.empty:
            return f"No patient record found for '{query}'."

        row = rows.iloc[0]
        summary = self._merge_summaries(rows) or "No summary available."
        return (
            f"Patient record found:\n"
            f"Name: {self._clean_value(row.get('Name'))}\n"
            f"Phone: {self._clean_value(row.get('Phone_number'))}\n"
            f"Email: {self._clean_value(row.get('Email')) or 'N/A'}\n"
            f"Age: {self._clean_value(row.get('Age'))}\n"
            f"Gender: {self._clean_value(row.get('Gender'))}\n"
            f"Address: {self._clean_value(row.get('Address'))}\n"
            f"Summary: {summary}"
        )

    def update_record(self, patient_name: str, new_note: str) -> str:
        self._reload()
        rows = self._match_rows(patient_name)
        if rows.empty:
            return f"No patient record found for '{patient_name}'."

        index = rows.index[0]
        existing = self._clean_value(self.df.loc[index, "Summary"])
        updated = (existing + " " + new_note).strip() if existing else new_note.strip()
        self.df.loc[index, "Summary"] = updated
        self.df.to_excel(RECORDS_FILE, index=False)
        return f"Record updated for {self._clean_value(self.df.loc[index, 'Name'])}."

    def summarize_history(self, patient_name: str) -> str:
        self._reload()
        rows = self._match_rows(patient_name)
        if rows.empty:
            return f"No patient record found for '{patient_name}'."

        row = rows.iloc[0]
        name = self._clean_value(row.get("Name"))
        summary = self._merge_summaries(rows) or "No structured summary in the spreadsheet."
        related_reports: List[str] = []

        for filename, text in self.pdf_texts.items():
            text_norm = normalize_text(text)
            if normalize_text(name) in text_norm or normalize_text(name.split()[0]) in text_norm:
                snippet = re.sub(r"\s+", " ", text)[:420]
                related_reports.append(f"{filename}: {snippet}...")

        report_section = "\n".join(related_reports) if related_reports else "No matching PDF report found."
        return (
            f"Medical history summary for {name}:\n"
            f"- Demographics: {self._clean_value(row.get('Age'))}-year-old {self._clean_value(row.get('Gender'))}.\n"
            f"- Address: {self._clean_value(row.get('Address'))}.\n"
            f"- Spreadsheet summary: {summary}\n"
            f"- Related reports:\n{report_section}"
        )
