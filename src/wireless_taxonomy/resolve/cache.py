from __future__ import annotations

import json
import sqlite3
from typing import Any

from wireless_taxonomy.models import utc_now


class SqliteResolverCache:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_candidates(self, provider: str, cache_key: str) -> list[tuple[str | None, str]] | None:
        row = self.conn.execute(
            "SELECT payload_json FROM resolver_cache WHERE provider = ? AND cache_key = ?",
            (provider, cache_key),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            return None
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(candidates, list):
            return None
        result: list[tuple[str | None, str]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            url = candidate.get("url")
            if not isinstance(url, str) or not url:
                continue
            label = candidate.get("label")
            result.append((str(label) if label is not None else None, url))
        return result

    def set_candidates(
        self,
        provider: str,
        cache_key: str,
        candidates: list[tuple[str | None, str]],
        status: str = "ok",
        error_message: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "candidates": [{"label": label, "url": url} for label, url in candidates],
        }
        self.conn.execute(
            """
            INSERT INTO resolver_cache(provider, cache_key, payload_json, status, error_message, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, cache_key) DO UPDATE SET
              payload_json = excluded.payload_json,
              status = excluded.status,
              error_message = excluded.error_message,
              fetched_at = excluded.fetched_at
            """,
            (provider, cache_key, json.dumps(payload, ensure_ascii=False), status, error_message, utc_now()),
        )
