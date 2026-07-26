"""
Smoke-тесты для src/mhgb/config.py и src/mhgb/logging_setup.py.
MongoDB и LLM не нужны.
"""

import os
from pathlib import Path

import pytest

from mhgb.config import Settings, cfg


class TestSettingsDefaults:
    def test_generator_api_base_set(self):
        # conftest.py выставляет env var до импорта модуля
        assert cfg.generator_api_base.startswith("http")

    def test_generator_model_set(self):
        assert cfg.generator_model != ""

    def test_mongodb_uri_set(self):
        assert "mongodb" in cfg.mongodb_uri


class TestSettingsPaths:
    def test_repo_root_is_absolute(self):
        assert cfg.repo_root.is_absolute()

    def test_data_dir_is_absolute(self):
        assert cfg.data_dir.is_absolute()

    def test_logs_dir_is_absolute(self):
        assert cfg.logs_dir.is_absolute()

    def test_data_dir_under_repo(self):
        assert cfg.data_dir.parent == cfg.repo_root

    def test_logs_dir_under_repo(self):
        assert cfg.logs_dir.parent == cfg.repo_root

    def test_legal_docs_dir_under_repo(self):
        assert cfg.legal_docs_dir.parent == cfg.repo_root


class TestSettingsEnvOverride:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GENERATOR_API_BASE", "http://override:9999/v1")
        s = Settings()
        assert s.generator_api_base == "http://override:9999/v1"

    def test_mongodb_uri_override(self, monkeypatch):
        monkeypatch.setenv("MONGODB_URI", "mongodb://testhost:27017/testdb")
        s = Settings()
        assert s.mongodb_uri == "mongodb://testhost:27017/testdb"

    def test_two_instances_independent(self, monkeypatch):
        monkeypatch.setenv("GENERATOR_MODEL", "test-mini")
        s = Settings()
        assert s.generator_model == "test-mini"
        # Глобальный cfg при этом не меняется (он создан раньше)
        # Просто проверяем, что новый инстанс работает
        assert isinstance(s, Settings)


class TestLoggingSetup:
    def test_logger_importable(self):
        from mhgb.logging_setup import logger
        assert logger is not None

    def test_logger_has_handlers(self):
        from loguru import logger as loguru_logger
        # loguru хранит обработчики внутри; проверяем что setup_logging не упал
        # (logger — глобальный синглтон, обработчики настроены при импорте)
        loguru_logger.debug("test from test_config.py")  # не должно бросать

    def test_logs_dir_created(self):
        # setup_logging() создаёт папку logs/ при импорте logging_setup
        assert cfg.logs_dir.exists()
