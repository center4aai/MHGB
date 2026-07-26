"""
Тесты для src/mhgb/models/llm_client.py.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from mhgb.models.llm_client import (
    GigaChatClient,
    LLMClient,
    OpenAICompatibleClient,
)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _openai_response(content: str) -> MagicMock:
    """Мок ответа openai.chat.completions.create."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


def _make_openai_client(content: str = "ok", **kwargs) -> tuple[OpenAICompatibleClient, MagicMock]:
    """Создаёт OpenAICompatibleClient с замоканным _OpenAI."""
    with patch("mhgb.models.llm_client._OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _openai_response(content)
        client = OpenAICompatibleClient(api_base="http://localhost/v1", model="qwen", **kwargs)
        # сохраняем мок-клиент внутри объекта после выхода из контекста
        client._client = MockOpenAI.return_value
        return client, MockOpenAI.return_value


def _gigachat_post_side_effects(
    content: str = "ok",
    token: str = "tok",
    expires_offset_sec: int = 3600,
) -> list[MagicMock]:
    """Возвращает [oauth_resp, chat_resp] для mock_post.side_effect."""
    oauth = MagicMock()
    oauth.raise_for_status = MagicMock()
    oauth.json.return_value = {
        "access_token": token,
        "expires_at": int((time.time() + expires_offset_sec) * 1000),
    }

    chat = MagicMock()
    chat.status_code = 200
    chat.raise_for_status = MagicMock()
    chat.json.return_value = {"choices": [{"message": {"content": content}}]}

    return [oauth, chat]


# ---------------------------------------------------------------------------
# OpenAICompatibleClient
# ---------------------------------------------------------------------------

class TestOpenAICompatibleClient:

    def test_empty_api_base_raises(self):
        with patch("mhgb.models.llm_client._OpenAI"):
            with pytest.raises(ValueError, match="api_base"):
                OpenAICompatibleClient(api_base="", model="qwen")

    def test_generate_returns_string(self):
        client, _ = _make_openai_client("response text")
        result = client.generate("sys", "user")
        assert result == "response text"

    def test_empty_system_prompt_not_in_messages(self):
        client, mock_inner = _make_openai_client()
        client.generate("", "hello")
        messages = mock_inner.chat.completions.create.call_args.kwargs["messages"]
        assert all(m["role"] != "system" for m in messages)
        assert messages[0]["content"] == "hello"

    def test_system_prompt_prepended(self):
        client, mock_inner = _make_openai_client()
        client.generate("система", "вопрос")
        messages = mock_inner.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "система"}
        assert messages[1] == {"role": "user",   "content": "вопрос"}

    def test_kwargs_override_temperature(self):
        client, mock_inner = _make_openai_client(temperature=0.0)
        client.generate("s", "u", temperature=0.9)
        assert mock_inner.chat.completions.create.call_args.kwargs["temperature"] == 0.9

    def test_kwargs_override_max_tokens(self):
        client, mock_inner = _make_openai_client(max_tokens=512)
        client.generate("s", "u", max_tokens=128)
        assert mock_inner.chat.completions.create.call_args.kwargs["max_tokens"] == 128

    def test_defaults_used_when_no_kwargs(self):
        client, mock_inner = _make_openai_client(temperature=0.3, max_tokens=777)
        client.generate("s", "u")
        kw = mock_inner.chat.completions.create.call_args.kwargs
        assert kw["temperature"] == 0.3
        assert kw["max_tokens"]  == 777

    def test_retry_succeeds_after_one_failure(self):
        with patch("mhgb.models.llm_client._OpenAI") as MockOpenAI:
            with patch("mhgb.models.llm_client.time.sleep"):
                mock_inner = MockOpenAI.return_value
                mock_inner.chat.completions.create.side_effect = [
                    RuntimeError("временная ошибка"),
                    _openai_response("success"),
                ]
                client = OpenAICompatibleClient(api_base="http://host/v1", model="m", retries=3)
                client._client = mock_inner
                result = client.generate("s", "u")
        assert result == "success"
        assert mock_inner.chat.completions.create.call_count == 2

    def test_retry_exhausted_raises_runtime_error(self):
        with patch("mhgb.models.llm_client._OpenAI") as MockOpenAI:
            with patch("mhgb.models.llm_client.time.sleep"):
                mock_inner = MockOpenAI.return_value
                mock_inner.chat.completions.create.side_effect = ConnectionError("нет связи")
                client = OpenAICompatibleClient(api_base="http://host/v1", model="m", retries=3)
                client._client = mock_inner
                with pytest.raises(RuntimeError, match="попыток исчерпаны"):
                    client.generate("s", "u")
        assert mock_inner.chat.completions.create.call_count == 3

    def test_none_content_returns_empty_string(self):
        client, mock_inner = _make_openai_client()
        mock_inner.chat.completions.create.return_value.choices[0].message.content = None
        result = client.generate("s", "u")
        assert result == ""

    def test_protocol_compatible(self):
        with patch("mhgb.models.llm_client._OpenAI"):
            client = OpenAICompatibleClient(api_base="http://host/v1", model="m")
        assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# GigaChatClient
# ---------------------------------------------------------------------------

class TestGigaChatClient:

    def test_empty_credentials_raises(self):
        with pytest.raises(ValueError, match="credentials"):
            GigaChatClient(credentials="")

    def test_token_fetched_on_first_generate(self):
        with patch("mhgb.models.llm_client.httpx.post") as mock_post:
            with patch("mhgb.models.llm_client.time.sleep"):
                mock_post.side_effect = _gigachat_post_side_effects("ответ")
                client = GigaChatClient(credentials="creds")
                client.generate("sys", "user")
        # первый вызов — OAuth, второй — chat
        assert mock_post.call_count == 2
        oauth_url = mock_post.call_args_list[0].args[0]
        assert "oauth" in oauth_url

    def test_token_not_refreshed_when_valid(self):
        with patch("mhgb.models.llm_client.httpx.post") as mock_post:
            with patch("mhgb.models.llm_client.time.sleep"):
                effects = _gigachat_post_side_effects()
                # второй chat-ответ для второго вызова generate
                chat2 = MagicMock()
                chat2.status_code = 200
                chat2.raise_for_status = MagicMock()
                chat2.json.return_value = {"choices": [{"message": {"content": "2"}}]}
                mock_post.side_effect = effects + [chat2]

                client = GigaChatClient(credentials="creds")
                client.generate("s", "u")  # OAuth + chat
                client.generate("s", "u")  # только chat (токен ещё живой)

        # oauth вызывается только один раз
        oauth_calls = [c for c in mock_post.call_args_list if "oauth" in c.args[0]]
        assert len(oauth_calls) == 1

    def test_token_refreshed_when_expired(self):
        with patch("mhgb.models.llm_client.httpx.post") as mock_post:
            with patch("mhgb.models.llm_client.time.sleep"):
                # expires_offset_sec=0 → токен сразу просрочен (минус 60с буфер)
                effects1 = _gigachat_post_side_effects(expires_offset_sec=0)
                effects2 = _gigachat_post_side_effects(content="second")
                mock_post.side_effect = effects1 + effects2

                client = GigaChatClient(credentials="creds")
                client.generate("s", "u")
                client.generate("s", "u")  # токен уже истёк → refresh

        oauth_calls = [c for c in mock_post.call_args_list if "oauth" in c.args[0]]
        assert len(oauth_calls) == 2

    def test_401_resets_token(self):
        with patch("mhgb.models.llm_client.httpx.post") as mock_post:
            with patch("mhgb.models.llm_client.time.sleep"):
                oauth_resp = MagicMock()
                oauth_resp.raise_for_status = MagicMock()
                oauth_resp.json.return_value = {
                    "access_token": "tok",
                    "expires_at": int((time.time() + 3600) * 1000),
                }

                chat_401 = MagicMock()
                chat_401.status_code = 401
                chat_401.raise_for_status.side_effect = Exception("401")

                mock_post.side_effect = [oauth_resp, chat_401, oauth_resp, chat_401, oauth_resp, chat_401]

                client = GigaChatClient(credentials="creds", retries=3)
                with pytest.raises(RuntimeError):
                    client.generate("s", "u")

        # после каждого 401 токен сбрасывается → OAuth вызывается перед каждой попыткой
        oauth_calls = [c for c in mock_post.call_args_list if "oauth" in c.args[0]]
        assert len(oauth_calls) == 3

    def test_generate_returns_string(self):
        with patch("mhgb.models.llm_client.httpx.post") as mock_post:
            mock_post.side_effect = _gigachat_post_side_effects("результат")
            client = GigaChatClient(credentials="creds")
            result = client.generate("sys", "user")
        assert result == "результат"

    def test_system_prompt_in_messages(self):
        with patch("mhgb.models.llm_client.httpx.post") as mock_post:
            mock_post.side_effect = _gigachat_post_side_effects()
            client = GigaChatClient(credentials="creds")
            client.generate("системный", "пользовательский")
        chat_call = mock_post.call_args_list[1]
        messages = chat_call.kwargs["json"]["messages"]
        assert messages[0] == {"role": "system", "content": "системный"}
        assert messages[1] == {"role": "user",   "content": "пользовательский"}

    def test_kwargs_override_temperature(self):
        with patch("mhgb.models.llm_client.httpx.post") as mock_post:
            mock_post.side_effect = _gigachat_post_side_effects()
            client = GigaChatClient(credentials="creds", temperature=0.0)
            client.generate("s", "u", temperature=0.7)
        payload = mock_post.call_args_list[1].kwargs["json"]
        assert payload["temperature"] == 0.7

    def test_retry_on_failure(self):
        with patch("mhgb.models.llm_client.httpx.post") as mock_post:
            with patch("mhgb.models.llm_client.time.sleep"):
                oauth = MagicMock()
                oauth.raise_for_status = MagicMock()
                oauth.json.return_value = {
                    "access_token": "tok",
                    "expires_at": int((time.time() + 3600) * 1000),
                }
                bad_chat = MagicMock()
                bad_chat.status_code = 500
                bad_chat.raise_for_status.side_effect = Exception("500")

                good_chat = MagicMock()
                good_chat.status_code = 200
                good_chat.raise_for_status = MagicMock()
                good_chat.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

                mock_post.side_effect = [oauth, bad_chat, oauth, good_chat]

                client = GigaChatClient(credentials="creds", retries=3)
                result = client.generate("s", "u")
        assert result == "ok"

    def test_retry_exhausted_raises_runtime_error(self):
        with patch("mhgb.models.llm_client.httpx.post") as mock_post:
            with patch("mhgb.models.llm_client.time.sleep"):
                oauth = MagicMock()
                oauth.raise_for_status = MagicMock()
                oauth.json.return_value = {
                    "access_token": "tok",
                    "expires_at": int((time.time() + 3600) * 1000),
                }
                bad = MagicMock()
                bad.status_code = 503
                bad.raise_for_status.side_effect = Exception("503")

                mock_post.side_effect = [oauth, bad] * 3

                client = GigaChatClient(credentials="creds", retries=3)
                with pytest.raises(RuntimeError, match="попыток исчерпаны"):
                    client.generate("s", "u")

    def test_protocol_compatible(self):
        client = GigaChatClient(credentials="creds")
        assert isinstance(client, LLMClient)
