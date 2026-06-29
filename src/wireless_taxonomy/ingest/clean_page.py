
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class CleanPage:
    source_url: str
    text: str
    links: list[tuple[str, str]]


class _TextAndLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.lines: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._link_text: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article", "br"}:
            self.lines.append("\n")
        if tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href")
            self._href = urljoin(self.base_url, href) if href else None
            self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "a" and self._href:
            text = _squash(" ".join(self._link_text))
            self.links.append((text, self._href))
            if text:
                self.lines.append(f" {text} [{self._href}] ")
            self._href = None
            self._link_text = []
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}:
            self.lines.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._href:
            self._link_text.append(data)
        else:
            self.lines.append(data)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_html(html: str, source_url: str) -> CleanPage:
    parser = _TextAndLinkParser(source_url)
    parser.feed(html)
    text = "\n".join(line for line in (_squash(part) for part in "".join(parser.lines).splitlines()) if line)
    return CleanPage(source_url=source_url, text=text, links=parser.links)


def fetch_clean_page(source_url: str) -> CleanPage:
    if source_url.startswith("file://"):
        path = Path(source_url.removeprefix("file://"))
        return clean_html(path.read_text(encoding="utf-8"), source_url)
    path = _resolve_local_path(source_url)
    if path.exists():
        return clean_html(path.read_text(encoding="utf-8"), path.resolve().as_uri())
    if "://" not in source_url:
        raise FileNotFoundError(
            f"Local input path not found: {source_url}. "
            "Use an absolute path, a path relative to the project root, or a full URL."
        )
    request = Request(source_url, headers={"User-Agent": "wireless-taxonomy/0.1"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    return clean_html(html, source_url)


def _resolve_local_path(value: str) -> Path:
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    # CLI users commonly run from src/ while passing paths relative to repo root.
    for parent in Path.cwd().resolve().parents:
        candidate = parent / value
        if candidate.exists():
            return candidate
    return path
