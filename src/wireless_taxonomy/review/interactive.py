from __future__ import annotations

import sqlite3

from wireless_taxonomy.review.queue import list_pending_review


def review_summary(conn: sqlite3.Connection, run_id: int | None = None) -> str:
    rows = list_pending_review(conn, run_id)
    if not rows:
        return "No pending review items."
    return "\n".join(
        f"#{row['id']} [{row['item_type']}] {row['field']}: {row['suggested_value'] or ''} "
        f"({row['confidence']:.2f}) - {row['review_reason']}"
        for row in rows
    )
