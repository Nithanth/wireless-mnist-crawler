from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wireless_taxonomy.textnorm import normalize_doi, normalize_title


@dataclass(frozen=True)
class GoldRecord:
    title: str
    venue: str
    year: int
    doi: str | None
    normalized_title: str
    normalized_doi: str
    raw: dict[str, Any] = field(default_factory=dict)


_TITLE_KEYS = ("paper title", "title", "name", "paper")
_DOI_KEYS = ("doi", "doi version of key", "doi url")
_VENUE_KEYS = ("conference", "venue", "conf")
_YEAR_KEYS = ("year", "yr")
_WIRELESS_KEYS = ("wireless", "is_wireless", "wireless?", "wireless candidate")
_TRUE_VALUES = {"1", "yes", "y", "true", "t", "wireless", "x"}


class GoldSheetReader:
    """Reads a manually curated gold sheet (csv/xlsx) into GoldRecords via pandas.

    Column names are matched case-insensitively against common variants so the
    user's own sheet layout does not need to be reshaped first.
    """

    def __init__(
        self,
        path: str,
        default_venue: str | None = None,
        default_year: int | None = None,
        wireless_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.default_venue = default_venue
        self.default_year = default_year
        self.wireless_only = wireless_only

    def read(self) -> list[GoldRecord]:
        import pandas as pd

        if self.path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(self.path)
        else:
            df = pd.read_csv(self.path)
        df.columns = [str(c).strip() for c in df.columns]
        lower_map = {c.lower(): c for c in df.columns}

        title_col = _find(lower_map, _TITLE_KEYS)
        if title_col is None:
            raise ValueError(
                f"Could not find a title column in {self.path.name}. "
                f"Columns seen: {list(df.columns)}"
            )
        doi_col = _find(lower_map, _DOI_KEYS)
        venue_col = _find(lower_map, _VENUE_KEYS)
        year_col = _find(lower_map, _YEAR_KEYS)
        wireless_col = _find(lower_map, _WIRELESS_KEYS)

        records: list[GoldRecord] = []
        for raw_row in df.to_dict("records"):
            title = _clean(raw_row.get(title_col))
            if not title:
                continue
            if self.wireless_only and wireless_col is not None:
                if _clean(raw_row.get(wireless_col)).lower() not in _TRUE_VALUES:
                    continue
            venue = _clean(raw_row.get(venue_col)) if venue_col else ""
            venue = venue or (self.default_venue or "")
            year = _coerce_year(raw_row.get(year_col) if year_col else None, self.default_year)
            if not venue or year is None:
                raise ValueError(
                    f"Row '{title[:60]}' is missing venue/year and no defaults were provided. "
                    "Pass --venue/--year or include conference/year columns."
                )
            doi = _clean(raw_row.get(doi_col)) if doi_col else ""
            records.append(
                GoldRecord(
                    title=title,
                    venue=venue,
                    year=year,
                    doi=doi or None,
                    normalized_title=normalize_title(title),
                    normalized_doi=normalize_doi(doi),
                    raw={str(k): _jsonable(v) for k, v in raw_row.items()},
                )
            )
        return records


def _find(lower_map: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in lower_map:
            return lower_map[key]
    return None


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def _coerce_year(value: Any, default: int | None) -> int | None:
    text = _clean(value)
    if text:
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return default
    return default


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)
