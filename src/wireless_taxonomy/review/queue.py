from __future__ import annotations

import sqlite3

from wireless_taxonomy.models import ReviewItem


def insert_review_item(conn: sqlite3.Connection, run_id: int | None, item: ReviewItem) -> None:
    conn.execute(
        """
        INSERT INTO review_items
        (run_id, item_type, paper_title, dataset_name, field, suggested_value, confidence,
         review_reason, evidence, source_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            item.item_type,
            item.paper_title,
            item.dataset_name,
            item.field,
            item.suggested_value,
            item.confidence,
            item.review_reason,
            item.evidence,
            item.source_url,
        ),
    )


def list_pending_review(conn: sqlite3.Connection, run_id: int | None = None) -> list[sqlite3.Row]:
    if run_id is None:
        return list(conn.execute("SELECT * FROM review_items WHERE status = 'pending' ORDER BY id"))
    return list(conn.execute("SELECT * FROM review_items WHERE status = 'pending' AND run_id = ? ORDER BY id", (run_id,)))
