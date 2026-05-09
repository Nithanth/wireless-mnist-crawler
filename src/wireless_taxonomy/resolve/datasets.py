from __future__ import annotations

import re
import sqlite3

from wireless_taxonomy.models import DatasetIdentityDecision


def normalize_dataset_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


class DatasetResolver:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def resolve(self, dataset_name: str) -> DatasetIdentityDecision:
        normalized = normalize_dataset_name(dataset_name)
        row = self.conn.execute("SELECT id FROM datasets WHERE normalized_name = ?", (normalized,)).fetchone()
        if row:
            return DatasetIdentityDecision(dataset_name, row["id"], "merge", 0.98, "Exact normalized dataset-name match")
        return DatasetIdentityDecision(dataset_name, None, "create", 0.99, "No existing normalized match")
