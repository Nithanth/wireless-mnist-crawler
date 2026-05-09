from __future__ import annotations

import sqlite3


def compute_reuse_counts(conn: sqlite3.Connection) -> dict[int, int]:
    rows = conn.execute(
        "SELECT dataset_id, COUNT(DISTINCT paper_id) AS count FROM paper_dataset_links GROUP BY dataset_id"
    )
    return {int(row["dataset_id"]): int(row["count"]) for row in rows}
