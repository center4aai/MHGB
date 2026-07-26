"""
Унифицированный LLM-клиент для экспериментов MHGB.

Два конкретных класса реализуют Protocol LLMClient:

  OpenAICompatibleClient — любой OpenAI-совместимый эндпоинт:
    • локальные серверы llama.cpp  (http://host:port/v1)
    • OpenAI API
    • YandexGPT / AliceLLM        (https://ai.api.cloud.yandex.net/v1)

  GigaChatClient — Сбер GigaChat с OAuth2-авторизацией:
    • автоматически получает и обновляет bearer-токен
    • тот же интерфейс generate(), что и у OpenAICompatibleClient
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
from openai import OpenAI as _OpenAI


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """Минимальный интерфейс для любого LLM-клиента."""

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Сгенерировать ответ на пару (system, user) промптов."""
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

@dataclass
class OpenAICompatibleClient:
    """
    Клиент для любого OpenAI-совместимого API.

    Примеры api_base:
      "http://localhost:8080/v1"                                — локальный llama.cpp
      "https://api.openai.com/v1"                                — OpenAI
      "https://ai.api.cloud.yandex.net/v1"                      — YandexGPT
    """

    api_base: str
    model: str
    api_key: str = "dummy"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 300.0
    retries: int = 3
    extra_body: dict | None = None   # доп. поля запроса (напр. {"reasoning_effort": "low"} для gpt-oss)

    def __post_init__(self) -> None:
        if not self.api_base:
            raise ValueError("api_base не может быть пустым")
        # connect_timeout=10с: если сервер упал → ошибка за 10с, не за 300с.
        # read_timeout = настраиваемый (300с по умолчанию) — для длинных reasoning-ответов.
        _timeout = httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=5.0)
        self._client = _OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=_timeout,
        )
        self.last_usage: dict = {}

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens  = kwargs.get("max_tokens",  self.max_tokens)

        create_kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body

        last_exc: BaseException | None = None
        for attempt in range(self.retries):
            try:
                resp = self._client.chat.completions.create(**create_kwargs)
                if resp.usage:
                    self.last_usage = {
                        "tokens_input":  resp.usage.prompt_tokens or 0,
                        "tokens_output": resp.usage.completion_tokens or 0,
                    }
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"OpenAICompatibleClient: {self.retries} попыток исчерпаны. "
            f"Последняя ошибка: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Native Ollama client — нужен, чтобы УПРАВЛЯТЬ num_ctx (OpenAI-эндпоинт его игнорит)
# ---------------------------------------------------------------------------

@dataclass
class NativeOllamaClient:
    """Клиент Ollama через нативный /api/chat — позволяет задавать num_ctx (окно).

    Нужен для СУДЕЙСТВА на Ollama (.64): OpenAI-эндпоинт /v1 игнорирует num_ctx и грузит
    модель на серверном дефолте 131072 (для 70B = ~98ГБ = вся карта → OOM/вытеснение
    чужих моделей на общем сервере). Native /api/chat уважает options.num_ctx.

    extra: top-level поля запроса (напр. {"reasoning_effort": "low"} для gpt-oss).
    api_base принимается в OpenAI-форме (.../v1) — /v1 отсекается, обращение к /api/chat.
    """

    api_base: str
    model: str
    num_ctx: int = 16384
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 300.0
    retries: int = 3
    extra: dict | None = None
    think: bool | None = None   # /api/chat top-level: False = отключить thinking-канал
                                # (gemma4 — рассуждение inline в content, иначе runaway/overflow)

    def __post_init__(self) -> None:
        if not self.api_base:
            raise ValueError("api_base не может быть пустым")
        base = self.api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self._url = base.rstrip("/") + "/api/chat"
        self.last_usage: dict = {}
        # Persistent client с keep-alive пулом: переиспользует TCP-соединение
        # (порт) между вызовами. Без него httpx.post открывал НОВЫЙ ephemeral-порт
        # на КАЖДЫЙ судейский вызов → при частом судействе быстрых моделей
        # (o3 ~10с/задача) порты Windows исчерпывались (TIME_WAIT ~4мин) →
        # массовый WinError 10048 (o3-барьер: 23/50 падений судьи).
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        if self.think is not None:
            body["think"] = self.think   # отключение thinking-канала (gemma4)
        if self.extra:
            body.update(self.extra)  # напр. reasoning_effort=low (top-level)

        last_exc: BaseException | None = None
        for attempt in range(self.retries):
            try:
                resp = self._client.post(self._url, json=body)
                resp.raise_for_status()
                data = resp.json()
                self.last_usage = {
                    "tokens_input":  data.get("prompt_eval_count", 0) or 0,
                    "tokens_output": data.get("eval_count", 0) or 0,
                }
                return (data.get("message", {}) or {}).get("content", "") or ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"NativeOllamaClient: {self.retries} попыток исчерпаны. Последняя ошибка: {last_exc}"
        )


# ---------------------------------------------------------------------------
# GigaChat client (Sberbank OAuth2)
# ---------------------------------------------------------------------------

@dataclass
class GigaChatClient:
    """
    Клиент для GigaChat (Сбер) с автоматическим OAuth2-токеном.

    credentials — base64(client_id:client_secret), берётся из личного кабинета Сбера.
    Токен действует 30 минут; клиент обновляет его автоматически.
    """

    credentials: str
    scope: str = "GIGACHAT_API_PERS"
    model: str = "GigaChat"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 300.0
    retries: int = 3

    _AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    _CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    def __post_init__(self) -> None:
        if not self.credentials:
            raise ValueError("credentials не может быть пустым")
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self.last_usage: dict = {}

    # -- токен --

    def _refresh_token(self) -> None:
        resp = httpx.post(
            self._AUTH_URL,
            headers={
                "Authorization": f"Basic {self.credentials}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": self.scope},
            verify=False,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        # expires_at приходит в миллисекундах; берём с запасом 60 сек
        self._token_expires_at = data["expires_at"] / 1000.0 - 60.0

    def _ensure_token(self) -> str:
        if not self._token or time.time() >= self._token_expires_at:
            self._refresh_token()
        return self._token  # type: ignore[return-value]

    # -- generate --

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        last_exc: BaseException | None = None
        for attempt in range(self.retries):
            try:
                token = self._ensure_token()
                resp = httpx.post(
                    self._CHAT_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                    verify=False,
                    timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=5.0),
                )
                if resp.status_code == 401:
                    self._token = None  # токен протух — обновим на следующей попытке
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                self.last_usage = {
                    "tokens_input":  usage.get("prompt_tokens", 0),
                    "tokens_output": usage.get("completion_tokens", 0),
                }
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"GigaChatClient: {self.retries} попыток исчерпаны. "
            f"Последняя ошибка: {last_exc}"
        )
