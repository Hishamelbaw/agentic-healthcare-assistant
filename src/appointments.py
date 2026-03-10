from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from .config import APPOINTMENTS_FILE
from .utils import load_json, normalize_text, save_json


class AppointmentManager:
    def __init__(self) -> None:
        self.default_schedule: Dict[str, List[str]] = {
            "Dr. Smith":  ["09:00", "10:00", "11:00", "14:00"],
            "Dr. Patel":  ["09:30", "11:30", "15:00"],
            "Dr. Lee":    ["08:30", "10:30", "13:30", "16:00"],
            "Dr. Sharma": ["09:00", "10:30", "14:00", "15:30"],
            "Dr. Chen":   ["08:00", "11:00", "13:00", "16:30"],
        }
        self.data = load_json(APPOINTMENTS_FILE, {"appointments": []})

    def _taken_slots(self, doctor: str, date_text: str) -> set:
        return {
            item["time"]
            for item in self.data["appointments"]
            if item["doctor"] == doctor
            and item["date"] == date_text
            and item.get("status", "active") == "active"
        }

    def book(
        self,
        patient_name: str,
        doctor: str,
        date_text: str,
        preferred_time: Optional[str] = None,
    ) -> str:
        # Fuzzy doctor matching: accept "Dr Smith" -> "Dr. Smith"
        matched_doctor = self._match_doctor(doctor)
        if matched_doctor is None:
            doctors = ", ".join(self.default_schedule)
            return (
                f"Doctor '{doctor}' not found. Available doctors: {doctors}.\n"
                f"Try: 'book appointment for <Name> with Dr. Smith tomorrow morning'."
            )

        available = list(self.default_schedule[matched_doctor])
        taken = self._taken_slots(matched_doctor, date_text)
        free_slots = [slot for slot in available if slot not in taken]

        if preferred_time:
            if preferred_time in free_slots:
                chosen = preferred_time
            else:
                return (
                    f"'{preferred_time}' is not available for {matched_doctor} on {date_text}.\n"
                    f"Free slots: {', '.join(free_slots) or 'none available'}."
                )
        else:
            if not free_slots:
                return f"No slots available for {matched_doctor} on {date_text}."
            chosen = free_slots[0]

        entry = {
            "id": len(self.data["appointments"]) + 1,
            "patient_name": patient_name,
            "doctor": matched_doctor,
            "date": date_text,
            "time": chosen,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        self.data["appointments"].append(entry)
        save_json(APPOINTMENTS_FILE, self.data)
        return (
            f"Appointment booked successfully.\n"
            f"  Patient : {patient_name}\n"
            f"  Doctor  : {matched_doctor}\n"
            f"  Date    : {date_text}\n"
            f"  Time    : {chosen}\n"
            f"  Ref #   : {entry['id']}"
        )

    def cancel(self, patient_name: str) -> str:
        needle = normalize_text(patient_name)
        cancelled = []
        for item in self.data["appointments"]:
            if normalize_text(item["patient_name"]) == needle and item.get("status") == "active":
                item["status"] = "cancelled"
                cancelled.append(
                    f"  Ref #{item['id']} — {item['doctor']} on {item['date']} at {item['time']}"
                )
        if not cancelled:
            return f"No active appointments found for '{patient_name}'."
        save_json(APPOINTMENTS_FILE, self.data)
        return f"Cancelled {len(cancelled)} appointment(s) for {patient_name}:\n" + "\n".join(cancelled)

    def list_all(self) -> str:
        active = [a for a in self.data["appointments"] if a.get("status", "active") == "active"]
        if not active:
            return "No appointments currently scheduled."
        lines = ["Upcoming Appointments", "=" * 40]
        for a in sorted(active, key=lambda x: (x["date"], x["time"])):
            lines.append(
                f"  #{a['id']:>3} | {a['date']} {a['time']} | {a['patient_name']:<20} | {a['doctor']}"
            )
        return "\n".join(lines)

    def _match_doctor(self, name: str) -> Optional[str]:
        """Fuzzy match: normalise punctuation and case."""
        needle = normalize_text(name)
        for key in self.default_schedule:
            if normalize_text(key) == needle:
                return key
            # Match "dr smith" -> "Dr. Smith"
            key_no_dot = normalize_text(key.replace(".", ""))
            needle_no_dot = needle.replace(".", "")
            if key_no_dot == needle_no_dot:
                return key
        return None
