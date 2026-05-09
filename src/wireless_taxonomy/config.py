from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


LlmProvider = Literal["openai", "anthropic", "google"]


@dataclass(frozen=True)
class Thresholds:
    wireless_inclusion: float = 0.90
    dataset_use: float = 0.90
    modality: float = 0.90
    osi_layer: float = 0.90
    availability: float = 0.95
    dataset_merge: float = 0.95


@dataclass(frozen=True)
class ProviderConfig:
    provider: LlmProvider
    model: str
    api_key_env: str
    api_key_configured: bool


@dataclass(frozen=True)
class LlmSettings:
    primary_provider: LlmProvider
    fallback_providers: tuple[LlmProvider, ...]
    providers: dict[LlmProvider, ProviderConfig]
    embedding_provider: LlmProvider
    embedding_model: str

    def ordered_providers(self) -> tuple[ProviderConfig, ...]:
        ordered = (self.primary_provider, *self.fallback_providers)
        seen: set[LlmProvider] = set()
        configs: list[ProviderConfig] = []
        for provider in ordered:
            if provider in seen:
                continue
            seen.add(provider)
            configs.append(self.providers[provider])
        return tuple(configs)


@dataclass(frozen=True)
class Settings:
    db_path: Path
    evidence_dir: Path
    llm: LlmSettings
    enable_web_search: bool
    enable_metadata_check: bool
    thresholds: Thresholds


def load_settings(db_path: str | Path = "taxonomy.sqlite") -> Settings:
    load_dotenv()
    db_path = Path(db_path)
    evidence_dir = os.getenv("WIRELESS_TAXONOMY_EVIDENCE_DIR")
    default_evidence_dir = db_path.parent / "evidence" if db_path.parent != Path(".") else Path("evidence")
    return Settings(
        db_path=db_path,
        evidence_dir=Path(evidence_dir) if evidence_dir else default_evidence_dir,
        llm=load_llm_settings(),
        enable_web_search=os.getenv("WIRELESS_TAXONOMY_ENABLE_WEB_SEARCH", "0") == "1",
        enable_metadata_check=os.getenv("WIRELESS_TAXONOMY_ENABLE_METADATA_CHECK", "0") == "1",
        thresholds=Thresholds(),
    )


def load_llm_settings() -> LlmSettings:
    primary = _provider(os.getenv("WIRELESS_TAXONOMY_LLM_PROVIDER", "openai"))
    fallbacks = tuple(
        _provider(value)
        for value in _csv_env("WIRELESS_TAXONOMY_LLM_FALLBACKS")
        if _provider(value) != primary
    )
    providers: dict[LlmProvider, ProviderConfig] = {
        "openai": ProviderConfig(
            provider="openai",
            model=os.getenv("WIRELESS_TAXONOMY_OPENAI_MODEL", "gpt-5.4-mini"),
            api_key_env="OPENAI_API_KEY",
            api_key_configured=bool(os.getenv("OPENAI_API_KEY")),
        ),
        "anthropic": ProviderConfig(
            provider="anthropic",
            model=os.getenv("WIRELESS_TAXONOMY_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            api_key_env="ANTHROPIC_API_KEY",
            api_key_configured=bool(os.getenv("ANTHROPIC_API_KEY")),
        ),
        "google": ProviderConfig(
            provider="google",
            model=os.getenv("WIRELESS_TAXONOMY_GOOGLE_MODEL", "gemini-3-flash-preview"),
            api_key_env="GEMINI_API_KEY",
            api_key_configured=bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        ),
    }
    embedding_provider = _provider(os.getenv("WIRELESS_TAXONOMY_EMBEDDING_PROVIDER", "openai"))
    return LlmSettings(
        primary_provider=primary,
        fallback_providers=fallbacks,
        providers=providers,
        embedding_provider=embedding_provider,
        embedding_model=os.getenv("WIRELESS_TAXONOMY_EMBEDDING_MODEL", "text-embedding-3-small"),
    )


def _csv_env(name: str) -> list[str]:
    return [part.strip() for part in os.getenv(name, "").split(",") if part.strip()]


def _provider(value: str) -> LlmProvider:
    normalized = value.strip().lower()
    if normalized not in {"openai", "anthropic", "google"}:
        raise ValueError(f"Unsupported LLM provider: {value}. Expected openai, anthropic, or google.")
    return normalized  # type: ignore[return-value]


def load_dotenv(path: str | Path = ".env") -> None:
    dotenv_path = _resolve_dotenv_path(Path(path))
    if dotenv_path is None:
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _resolve_dotenv_path(path: Path) -> Path | None:
    if path.exists():
        return path
    for parent in Path.cwd().resolve().parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return None
