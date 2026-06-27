from pathlib import Path

import pytest

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.llm import LlmRouter


def test_llm_provider_order_and_key_detection(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("WIRELESS_TAXONOMY_LLM_FALLBACKS", "openai,google")
    monkeypatch.setenv("WIRELESS_TAXONOMY_ANTHROPIC_MODEL", "claude-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    settings = load_settings(tmp_path / "taxonomy.sqlite")
    ordered = settings.llm.ordered_providers()

    assert [provider.provider for provider in ordered] == ["anthropic", "openai", "google"]
    assert ordered[0].model == "claude-test"
    assert ordered[0].api_key_configured is True
    assert LlmRouter(settings.llm).select_provider().provider == "anthropic"


def test_default_model_names_are_current_requested_defaults(monkeypatch, tmp_path: Path) -> None:
    for name in [
        "WIRELESS_TAXONOMY_LLM_PROVIDER",
        "WIRELESS_TAXONOMY_LLM_FALLBACKS",
        "WIRELESS_TAXONOMY_OPENAI_MODEL",
        "WIRELESS_TAXONOMY_ANTHROPIC_MODEL",
        "WIRELESS_TAXONOMY_GOOGLE_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = load_settings(tmp_path / "taxonomy.sqlite")

    assert settings.llm.providers["openai"].model == "gpt-5.4-mini"
    assert settings.llm.providers["anthropic"].model == "claude-sonnet-4-6"
    assert settings.llm.providers["google"].model == "gemini-3.5-flash"


def test_google_provider_accepts_gemini_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    settings = load_settings(tmp_path / "taxonomy.sqlite")

    assert settings.llm.providers["google"].api_key_configured is True


def test_dotenv_is_loaded_from_working_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WIRELESS_TAXONOMY_LLM_PROVIDER", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("WIRELESS_TAXONOMY_LLM_PROVIDER=google\nGEMINI_API_KEY=secret\n", encoding="utf-8")

    settings = load_settings(tmp_path / "taxonomy.sqlite")

    assert settings.llm.primary_provider == "google"
    assert settings.llm.providers["google"].api_key_configured is True


def test_invalid_llm_provider_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_LLM_PROVIDER", "invalid")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        load_settings(tmp_path / "taxonomy.sqlite")
