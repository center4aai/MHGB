"""
Централизованная конфигурация MHGB.

Читает переменные из .env (если файл есть) и из окружения.
Использование:
    from mhgb.config import cfg
    client = OpenAI(base_url=cfg.generator_api_base)
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM-генератор (Qwen3 / llama.cpp)
    generator_api_base: str = "http://localhost:1234/v1"
    generator_model: str = "qwen3-6-27b"

    # MongoDB (optional — used only with --save-mongo)
    mongodb_uri: str = "mongodb://localhost:27017/mhgb"

    # ---------------------------------------------------------------------------
    # Вычисляемые пути (readonly properties — не читаются из .env)
    # ---------------------------------------------------------------------------

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def data_dir(self) -> Path:
        return REPO_ROOT / "data"

    @property
    def legal_docs_dir(self) -> Path:
        return REPO_ROOT / "legal_docs"

    @property
    def logs_dir(self) -> Path:
        return REPO_ROOT / "logs"


cfg = Settings()
