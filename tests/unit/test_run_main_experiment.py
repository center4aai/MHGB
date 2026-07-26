"""
Тесты для src/mhgb/experiments/run_main_experiment.py.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from mhgb.experiments.run_main_experiment import (
    _JudgeAdapter,
    _build_context,
    _is_done,
    _make_prompt,
    instantiate_client,
    load_models_config,
    run_experiment,
    run_one,
)


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------

def _model_cfg(
    name="test-model",
    client_type="openai_compatible",
    api_base="http://host/v1",
    model_id="m",
    **kwargs,
) -> dict:
    base = dict(
        name=name,
        client_type=client_type,
        api_base=api_base,
        model_id=model_id,
        max_tokens=512,
        temperature=0.0,
    )
    base.update(kwargs)
    return base


def _task(task_id="t1", mode="closed") -> dict:
    return {
        "id": task_id,
        "type": "rule_selection",
        "mode": mode,
        "hop_group": "shallow",
        "norm_ids": ["ТК_261"],
        "gold_chain": [{"step": 1, "norm_id": "ТК_261", "reasoning": "...", "conclusion": "..."}],
        "fabula": "Фабула задачи",
        "question": "Вопрос?",
        "answer": "Ответ",
        "expected_answer_format": "",
    }


def _eval_result() -> dict:
    return {
        "parsed":                   {"applicable_articles": "ТК_261", "reasoning": "...", "conclusion": "ok"},
        "norm_coverage_list":       0.8,
        "norm_coverage_reasoning":  0.7,
        "step_correctness":         0.9,
        "step_scores":              [1.0],
        "answer_correctness":       1.0,
        "final_score":              0.9,
        "judge_logs":               [],
    }


# ---------------------------------------------------------------------------
# load_models_config
# ---------------------------------------------------------------------------

class TestLoadModelsConfig:

    def test_loads_all_models(self, tmp_path):
        cfg = {"models": [_model_cfg("m1"), _model_cfg("m2")]}
        p = tmp_path / "models.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        result = load_models_config(str(p))
        assert len(result) == 2

    def test_model_has_name_field(self, tmp_path):
        cfg = {"models": [_model_cfg("qwen3-27b")]}
        p = tmp_path / "models.yaml"
        p.write_text(yaml.dump(cfg), encoding="utf-8")
        result = load_models_config(str(p))
        assert result[0]["name"] == "qwen3-27b"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_models_config("/nonexistent/path/models.yaml")


# ---------------------------------------------------------------------------
# instantiate_client
# ---------------------------------------------------------------------------

class TestInstantiateClient:

    def test_openai_compatible_returns_client(self):
        with patch("mhgb.models.llm_client._OpenAI"):
            from mhgb.models.llm_client import OpenAICompatibleClient
            client = instantiate_client(_model_cfg())
        assert isinstance(client, OpenAICompatibleClient)

    def test_gigachat_returns_client(self, monkeypatch):
        monkeypatch.setenv("GIGACHAT_CREDENTIALS", "base64creds")
        from mhgb.models.llm_client import GigaChatClient
        cfg = _model_cfg(client_type="gigachat", api_base=None)
        cfg["credentials_env"] = "GIGACHAT_CREDENTIALS"
        client = instantiate_client(cfg)
        assert isinstance(client, GigaChatClient)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="client_type"):
            instantiate_client(_model_cfg(client_type="unknown"))

    def test_missing_gigachat_env_raises(self, monkeypatch):
        monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
        cfg = _model_cfg(client_type="gigachat", api_base=None)
        cfg["credentials_env"] = "GIGACHAT_CREDENTIALS"
        with pytest.raises(ValueError, match="GIGACHAT_CREDENTIALS"):
            instantiate_client(cfg)


# ---------------------------------------------------------------------------
# _JudgeAdapter
# ---------------------------------------------------------------------------

class TestJudgeAdapter:

    def test_complete_calls_generate(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = "judge answer"
        adapter = _JudgeAdapter(mock_client)
        adapter.complete("sys", "user")
        mock_client.generate.assert_called_once_with("sys", "user")

    def test_complete_returns_response(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = "result"
        assert _JudgeAdapter(mock_client).complete("s", "u") == "result"


# ---------------------------------------------------------------------------
# _is_done
# ---------------------------------------------------------------------------

class TestIsDone:

    def test_returns_false_when_no_storage(self):
        assert _is_done("exp", "t1", "m1", "closed", None) is False

    def test_returns_true_when_found(self):
        storage = MagicMock()
        storage.db["model_responses"].count_documents.return_value = 1
        assert _is_done("exp", "t1", "m1", "closed", storage) is True

    def test_returns_false_when_not_found(self):
        storage = MagicMock()
        storage.db["model_responses"].count_documents.return_value = 0
        assert _is_done("exp", "t1", "m1", "closed", storage) is False

    def test_query_includes_all_four_fields(self):
        storage = MagicMock()
        storage.db["model_responses"].count_documents.return_value = 0
        _is_done("my_exp", "task_42", "qwen3", "open_full_graph", storage)
        query = storage.db["model_responses"].count_documents.call_args[0][0]
        assert query["experiment_name"] == "my_exp"
        assert query["task_id"]         == "task_42"
        assert query["model_name"]      == "qwen3"
        assert query["mode"]            == "open_full_graph"


# ---------------------------------------------------------------------------
# _make_prompt
# ---------------------------------------------------------------------------

class TestMakePrompt:

    def test_closed_no_context_section(self):
        prompt = _make_prompt(_task(), context_chunks=None)
        assert "Фабула:" in prompt
        assert "Вопрос:" in prompt
        assert "Контекст:" not in prompt

    def test_open_includes_context(self):
        chunk = MagicMock()
        chunk.norm_id = "ТК_261"
        chunk.title = "Статья 261"
        chunk.text = "Запрещается увольнение..."
        prompt = _make_prompt(_task(), context_chunks=[chunk])
        assert "Контекст:" in prompt
        assert "ТК_261" in prompt

    def test_answer_format_appended_when_present(self):
        task = _task()
        task["expected_answer_format"] = "ПРИМЕНИМЫЕ СТАТЬИ: ..."
        prompt = _make_prompt(task, context_chunks=None)
        assert "Формат ответа:" in prompt

    def test_answer_format_omitted_when_empty(self):
        task = _task()
        task["expected_answer_format"] = ""
        prompt = _make_prompt(task, context_chunks=None)
        assert "Формат ответа:" not in prompt


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

class TestBuildContext:

    def test_closed_returns_none(self):
        assert _build_context(_task(), "closed", MagicMock(), {}) is None

    def test_open_full_graph_calls_builder(self):
        with patch("mhgb.experiments.run_main_experiment.build_full_graph_context") as mock_build:
            mock_build.return_value = []
            result = _build_context(_task(), "open_full_graph", MagicMock(), {})
        mock_build.assert_called_once()
        assert result == []

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="режим"):
            _build_context(_task(), "unknown_mode", MagicMock(), {})


# ---------------------------------------------------------------------------
# run_one
# ---------------------------------------------------------------------------

class TestRunOne:

    def _run(self, dry_run=False, storage=None, generate_raises=False):
        mock_client = MagicMock()
        if generate_raises:
            mock_client.generate.side_effect = RuntimeError("сеть упала")
        else:
            mock_client.generate.return_value = "ПРИМЕНИМЫЕ СТАТЬИ: ТК_261\nЦЕПОЧКА РАССУЖДЕНИЙ: ok\nВЫВОД: нет"

        mock_judge = MagicMock()

        with patch("mhgb.experiments.run_main_experiment._evaluate", return_value=_eval_result()):
            with patch("mhgb.experiments.run_main_experiment._build_context", return_value=None):
                return run_one(
                    task=_task(),
                    model_name="qwen3-27b",
                    client=mock_client,
                    judge=mock_judge,
                    mode="closed",
                    graph=MagicMock(),
                    corpus={},
                    experiment_name="test_exp",
                    storage=storage,
                    dry_run=dry_run,
                )

    def test_dry_run_returns_result_dict(self):
        result = self._run(dry_run=True)
        assert result["model_name"] == "qwen3-27b"
        assert result["final_score"] == 0.9
        assert "error" not in result

    def test_dry_run_not_saved_to_mongo(self):
        storage = MagicMock()
        self._run(dry_run=True, storage=storage)
        storage.db.__getitem__.assert_not_called()

    def test_generate_error_returns_error_dict(self):
        result = self._run(generate_raises=True)
        assert "error" in result
        assert "сеть упала" in result["error"]
        assert result["task_id"] == "t1"

    def test_result_has_required_keys(self):
        result = self._run(dry_run=True)
        for key in ("experiment_name", "task_id", "model_name", "mode",
                    "final_score", "norm_coverage_list", "step_correctness"):
            assert key in result

    def test_saves_to_mongo_when_storage_provided(self):
        storage = MagicMock()
        self._run(storage=storage)
        assert storage.db["model_responses"].insert_one.called
        assert storage.db["evaluations"].insert_one.called


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

class TestRunExperiment:

    def _make_inputs(self, n_tasks=2, n_models=2, modes=None):
        tasks = [_task(f"task_{i}") for i in range(n_tasks)]
        models = [_model_cfg(f"model_{i}") for i in range(n_models)]
        modes = modes or ["closed"]
        return tasks, models, modes

    def _run_exp(self, tasks, models_cfg, modes, storage=None, models_filter=None):
        judge = MagicMock()
        judge.generate.return_value = "ПРИМЕНИМЫЕ СТАТЬИ: ТК_261\nЦЕПОЧКА РАССУЖДЕНИЙ: ok\nВЫВОД: нет"

        with patch("mhgb.experiments.run_main_experiment.instantiate_client") as mock_inst:
            mock_client = MagicMock()
            mock_client.generate.return_value = "response"
            mock_inst.return_value = mock_client

            with patch("mhgb.experiments.run_main_experiment._evaluate", return_value=_eval_result()):
                with patch("mhgb.experiments.run_main_experiment._build_context", return_value=None):
                    with patch("mhgb.experiments.run_main_experiment._is_done", return_value=False):
                        return run_experiment(
                            tasks=tasks,
                            models_cfg=models_cfg,
                            judge_client=judge,
                            graph=MagicMock(),
                            corpus={},
                            modes=modes,
                            experiment_name="test_exp",
                            storage=storage,
                            dry_run=True,
                            models_filter=models_filter,
                        )

    def test_returns_correct_count(self):
        tasks, models, modes = self._make_inputs(n_tasks=3, n_models=2, modes=["closed"])
        results = self._run_exp(tasks, models, modes)
        assert len(results) == 6  # 3 задачи × 2 модели × 1 режим

    def test_filters_models_by_name(self):
        tasks, models, modes = self._make_inputs(n_tasks=2, n_models=3)
        results = self._run_exp(tasks, models, modes, models_filter=["model_0"])
        assert all(r["model_name"] == "model_0" for r in results)

    def test_skips_already_done(self):
        tasks, models, modes = self._make_inputs(n_tasks=2, n_models=1)
        judge = MagicMock()
        judge.generate.return_value = "resp"

        with patch("mhgb.experiments.run_main_experiment.instantiate_client"):
            with patch("mhgb.experiments.run_main_experiment._evaluate", return_value=_eval_result()):
                with patch("mhgb.experiments.run_main_experiment._build_context", return_value=None):
                    with patch("mhgb.experiments.run_main_experiment._is_done", return_value=True):
                        results = run_experiment(
                            tasks=tasks, models_cfg=models, judge_client=judge,
                            graph=MagicMock(), corpus={}, modes=modes,
                            experiment_name="exp", dry_run=True,
                        )
        assert len(results) == 0

    def test_client_error_skips_model(self):
        tasks, models, modes = self._make_inputs(n_tasks=2, n_models=2)
        judge = MagicMock()

        with patch("mhgb.experiments.run_main_experiment.instantiate_client") as mock_inst:
            mock_inst.side_effect = [ValueError("нет ключа"), MagicMock()]
            with patch("mhgb.experiments.run_main_experiment._evaluate", return_value=_eval_result()):
                with patch("mhgb.experiments.run_main_experiment._build_context", return_value=None):
                    with patch("mhgb.experiments.run_main_experiment._is_done", return_value=False):
                        results = run_experiment(
                            tasks=tasks, models_cfg=models, judge_client=judge,
                            graph=MagicMock(), corpus={}, modes=modes,
                            experiment_name="exp", dry_run=True,
                        )
        # только model_1 обработан (model_0 пропущен из-за ошибки клиента)
        assert all(r["model_name"] == "model_1" for r in results)

    def test_two_modes_double_results(self):
        tasks, models, modes = self._make_inputs(n_tasks=2, n_models=1, modes=["closed", "open_full_graph"])
        results = self._run_exp(tasks, models, modes)
        assert len(results) == 4  # 2 задачи × 1 модель × 2 режима
