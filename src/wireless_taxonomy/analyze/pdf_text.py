from __future__ import annotations

import io
import re


def extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        return text or fallback_pdf_text(data)
    except Exception:
        return fallback_pdf_text(data)


def fallback_pdf_text(data: bytes) -> str:
    raw = data.decode("latin-1", errors="ignore")
    strings = re.findall(r"\(([^()]*)\)\s*Tj", raw)
    if not strings:
        strings = re.findall(r"\(([^()]*)\)", raw)
    cleaned = [_unescape_pdf_string(value) for value in strings if value.strip()]
    if cleaned:
        return "\n".join(cleaned).strip()
    return raw if "%PDF" not in raw[:20] else ""


def _unescape_pdf_string(value: str) -> str:
    return value.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
