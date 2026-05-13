from __future__ import annotations

import base64
import os
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from wireless_taxonomy.analyze.full_text import _artifact, _snippets_from_artifacts, extract_pdf_text
from wireless_taxonomy.models import PaperTextEnrichment, PaperTextLink


class AcmBrowserDependencyError(RuntimeError):
    pass


_BROWSER_FETCH_PDF_JS = """async (url) => {
  const response = await fetch(url, {credentials: 'include', headers: {'Accept': 'application/pdf'}});
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return {
    status: response.status,
    url: response.url,
    contentType: response.headers.get('content-type') || '',
    body: btoa(binary)
  };
}"""


class AuthenticatedAcmBrowserFetcher:
    provider_name = "acm_browser_authenticated_v0"

    def __init__(
        self,
        profile_dir: str | Path,
        headless: bool = False,
        browser_channel: str | None = None,
        cdp_url: str | None = None,
        delay_seconds: float | None = None,
        timeout_seconds: int = 60,
    ):
        self.profile_dir = Path(profile_dir)
        self.headless = headless
        self.browser_channel = browser_channel
        self.cdp_url = cdp_url
        self.delay_seconds = delay_seconds if delay_seconds is not None else _acm_browser_delay_seconds()
        self.timeout_seconds = timeout_seconds

    def login(self, login_url: str = "https://dl.acm.org/") -> None:
        with _playwright_context(self.profile_dir, self.headless, self.browser_channel, self.cdp_url) as (_, context):
            page = context.new_page()
            page.goto(login_url, wait_until="domcontentloaded", timeout=self.timeout_seconds * 1000)
            input("Log in through ACM/institutional access in the opened browser, then press Enter here to continue...")
            context.storage_state()

    def fetch_many(self, papers: Iterable[Mapping[str, Any]], limit: int | None = None) -> list[PaperTextEnrichment]:
        selected = list(papers)
        if limit is not None:
            selected = selected[:limit]
        enrichments: list[PaperTextEnrichment] = []
        with _playwright_context(self.profile_dir, self.headless, self.browser_channel, self.cdp_url) as (_, context):
            for index, paper in enumerate(selected):
                if index:
                    time.sleep(self.delay_seconds)
                enrichments.append(self._fetch_one(context, paper))
        return enrichments

    def _fetch_one(self, context: Any, paper: Mapping[str, Any]) -> PaperTextEnrichment:
        paper_id = int(paper["id"])
        pdf_url = _acm_pdf_url(paper)
        if not pdf_url:
            artifact = _artifact(paper_id, "acm_browser_pdf_text", None, "error", "", "No ACM DOI PDF URL available")
            return PaperTextEnrichment(paper_id, [artifact], [], [])
        try:
            status, final_url, content_type, data = self._fetch_pdf_bytes(context, pdf_url)
            if status in {401, 403}:
                artifact = _artifact(paper_id, "acm_browser_pdf_text", final_url, "error", "", f"ACM browser request returned HTTP {status}; login/access may be required")
            elif status >= 400:
                artifact = _artifact(paper_id, "acm_browser_pdf_text", final_url, "error", "", f"ACM browser request returned HTTP {status}")
            elif _looks_like_html(data, content_type):
                artifact = _artifact(paper_id, "acm_browser_pdf_text", final_url, "error", "", "ACM browser request returned HTML instead of PDF bytes")
            else:
                text = extract_pdf_text(data)
                artifact = _artifact(
                    paper_id,
                    "acm_browser_pdf_text",
                    final_url,
                    "fetched" if text.strip() else "empty",
                    text,
                    None if text.strip() else "No text extracted from ACM PDF",
                )
        except Exception as exc:
            artifact = _artifact(paper_id, "acm_browser_pdf_text", pdf_url, "error", "", str(exc))
        links = [PaperTextLink(paper_id, pdf_url, "acm_browser_pdf", "pdf", 0.95)]
        return PaperTextEnrichment(paper_id, [artifact], links, _snippets_from_artifacts(paper_id, [artifact]))

    def _fetch_pdf_bytes(self, context: Any, pdf_url: str) -> tuple[int, str, str, bytes]:
        # ACM/Cloudflare may reject Playwright's APIRequestContext. Navigating a tab
        # to a PDF can expose Chrome's PDF viewer HTML, so fetch bytes from a
        # dl.acm.org page where the browser's authenticated cookies are available.
        page = context.new_page()
        try:
            page.goto("https://dl.acm.org/", wait_until="domcontentloaded", timeout=self.timeout_seconds * 1000)
            result = page.evaluate(_BROWSER_FETCH_PDF_JS, pdf_url)
            data = base64.b64decode(str(result.get("body") or ""))
            return (
                int(result.get("status") or 0),
                str(result.get("url") or pdf_url),
                str(result.get("contentType") or "").lower(),
                data,
            )
        except Exception:
            response = page.goto(pdf_url, wait_until="domcontentloaded", timeout=self.timeout_seconds * 1000)
            if response is None:
                return 0, pdf_url, "", b""
            return (
                int(response.status),
                str(response.url or pdf_url),
                str(response.headers.get("content-type") or "").lower(),
                response.body(),
            )
        finally:
            page.close()


def _acm_pdf_url(paper: Mapping[str, Any]) -> str | None:
    doi = _optional(paper.get("doi"))
    if doi and doi.startswith("10.1145/"):
        return f"https://dl.acm.org/doi/pdf/{doi}"
    pdf_url = _optional(paper.get("pdf_url"))
    if pdf_url and _is_acm_pdf_url(pdf_url):
        return pdf_url
    paper_url = _optional(paper.get("paper_url"))
    if paper_url and "dl.acm.org/doi/" in paper_url:
        doi_part = paper_url.rstrip("/").rsplit("/doi/", 1)[-1]
        if doi_part:
            return f"https://dl.acm.org/doi/pdf/{doi_part}"
    return None


def _playwright_context(profile_dir: Path, headless: bool, browser_channel: str | None, cdp_url: str | None = None):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise AcmBrowserDependencyError(
            "Playwright is required for authenticated ACM browser fetch. "
            "Install it with: python3 -m pip install playwright && python3 -m playwright install chromium"
        ) from exc

    class _ContextManager:
        def __enter__(self):
            profile_dir.mkdir(parents=True, exist_ok=True)
            self._playwright = sync_playwright().start()
            self._browser = None
            if cdp_url:
                self._browser = self._playwright.chromium.connect_over_cdp(cdp_url)
                contexts = self._browser.contexts
                self._context = contexts[0] if contexts else self._browser.new_context(accept_downloads=True)
                return self._playwright, self._context
            kwargs: dict[str, Any] = {
                "user_data_dir": str(profile_dir),
                "headless": headless,
                "accept_downloads": True,
            }
            if browser_channel:
                kwargs["channel"] = browser_channel
            self._context = self._playwright.chromium.launch_persistent_context(**kwargs)
            return self._playwright, self._context

        def __exit__(self, exc_type, exc, tb):
            try:
                self._context.close()
            except Exception:
                pass
            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception:
                    pass
            self._playwright.stop()

    return _ContextManager()


def _acm_browser_delay_seconds() -> float:
    raw_value = os.getenv("WIRELESS_TAXONOMY_ACM_BROWSER_DELAY_SECONDS")
    if raw_value:
        try:
            return max(3.0, float(raw_value))
        except ValueError:
            pass
    return 8.0


def _is_acm_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.lower().endswith("dl.acm.org") and "/doi/pdf/" in parsed.path.lower()


def _looks_like_html(data: bytes, content_type: str) -> bool:
    head = data[:500].lstrip().lower()
    return "text/html" in content_type or head.startswith((b"<!doctype html", b"<html")) or bool(re.search(br"<html[\s>]", head[:200]))


def _optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
