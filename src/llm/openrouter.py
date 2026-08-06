"""Minimal OpenRouter chat client."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from examples.dsl_smoke_test import load_dotenv


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class OpenRouterUsage:
    """Token usage returned by OpenRouter."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def call_openrouter_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1,
    timeout_seconds: int = 60,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Call OpenRouter chat completions and return parsed JSON."""

    load_dotenv()
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment or .env")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/factor-lab",
            "X-Title": "Factor Lab",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
    return json.loads(raw_response)


def openrouter_usage(data: dict[str, Any]) -> OpenRouterUsage:
    """Parse token usage from OpenRouter response JSON."""

    usage = data.get("usage") or {}
    return OpenRouterUsage(
        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(usage.get("completion_tokens")),
        total_tokens=_int_or_none(usage.get("total_tokens")),
    )


def openrouter_content(data: dict[str, Any]) -> str:
    """Extract choices[0].message.content from OpenRouter response JSON."""

    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter response did not contain choices[0].message.content") from exc


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    return int(value)
