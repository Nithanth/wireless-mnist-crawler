from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import EvidenceClaim, new_id, utc_now


class EvidenceLogger:
    def __init__(self, evidence_dir: Path, run_id: int | None):
        self.evidence_dir = evidence_dir
        self.run_id = run_id
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"run_{run_id}" if run_id is not None else "unassigned"
        self.path = self.evidence_dir / f"{suffix}.jsonl"

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "event_id": new_id("evt"),
            "run_id": self.run_id,
            "event_type": event_type,
            "created_at": utc_now(),
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def claim(self, claim: EvidenceClaim) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(claim), ensure_ascii=False, sort_keys=True) + "\n")
