"""
Consent registry + audit log — the gate in front of every identity operation.

DPDP Act 2023 §9 requires verifiable parental consent before processing a
child's personal data; face embeddings are biometric. Rules enforced here:
  - A student is identifiable only with status "granted" AND the purpose
    requested (e.g. "engagement", "exam") in their purposes.
  - "pending" or "withdrawn" → treated exactly like unknown: blurred, class
    aggregates only, embeddings never stored.
  - Every enrollment, identification run, view and purge is appended to an
    audit log (JSONL, append-only by convention).

Registry CSV columns:
  student_id, name, section, status, purposes, verified_by, verified_on
  purposes is ";"-separated, e.g. "engagement;exam"
"""

import csv
import getpass
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

STATUSES = ("granted", "pending", "withdrawn")
PURPOSES = ("engagement", "exam")
FIELDS = ["student_id", "name", "section", "status", "purposes", "verified_by", "verified_on"]


@dataclass
class ConsentRecord:
    student_id: str
    name: str
    section: str
    status: str = "pending"
    purposes: list[str] = field(default_factory=list)
    verified_by: str = ""     # how guardian consent was verified (e.g. "signed form", "DigiLocker")
    verified_on: str = ""     # ISO date

    def allows(self, purpose: str) -> bool:
        return self.status == "granted" and purpose in self.purposes


class ConsentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: dict[str, ConsentRecord] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                status = row.get("status", "pending").strip().lower()
                if status not in STATUSES:
                    raise ValueError(f"{row.get('student_id')}: unknown consent status '{status}'")
                self._records[row["student_id"].strip()] = ConsentRecord(
                    student_id=row["student_id"].strip(),
                    name=row.get("name", "").strip(),
                    section=row.get("section", "").strip(),
                    status=status,
                    purposes=[p.strip() for p in row.get("purposes", "").split(";") if p.strip()],
                    verified_by=row.get("verified_by", "").strip(),
                    verified_on=row.get("verified_on", "").strip(),
                )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for r in self._records.values():
                writer.writerow({**asdict(r), "purposes": ";".join(r.purposes)})

    def get(self, student_id: str) -> Optional[ConsentRecord]:
        return self._records.get(student_id)

    def allows(self, student_id: str, purpose: str) -> bool:
        record = self._records.get(student_id)
        return record is not None and record.allows(purpose)

    def allowed_students(self, purpose: str) -> set[str]:
        return {sid for sid, r in self._records.items() if r.allows(purpose)}

    def records(self) -> list[ConsentRecord]:
        return list(self._records.values())

    def withdraw(self, student_id: str) -> None:
        record = self._records.get(student_id)
        if record is not None:
            record.status = "withdrawn"
            record.purposes = []

    def __len__(self) -> int:
        return len(self._records)


class AuditLog:
    """Append-only JSONL log of identity-related actions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, action: str, student_id: Optional[str] = None, **detail) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "actor": _actor(),
            "action": action,
            "student_id": student_id,
            **detail,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _actor() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"
