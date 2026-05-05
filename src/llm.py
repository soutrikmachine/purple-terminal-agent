"""
LLM client — OpenRouter via OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

MODEL    = os.environ.get("MODEL", "deepseek/deepseek-v4-flash")
API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", MODEL)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def complete(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.2,
    model_override: str | None = None,
) -> str:
    client = get_client()
    full_messages = [{"role": "system", "content": system}] + messages
    response = await client.chat.completions.create(
        model=model_override or MODEL,
        messages=full_messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = response.choices[0].message.content or ""
    logger.debug("complete [%d tok]: %s", max_tokens, result[:200])
    return result.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def complete_json(
    system: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.2,
    model_override: str | None = None,
) -> dict:
    client = get_client()
    full_messages = [{"role": "system", "content": system}] + messages
    response = await client.chat.completions.create(
        model=model_override or MODEL,
        messages=full_messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    clean = re.sub(r"```json\s*|\s*```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        logger.error("Failed to parse JSON: %s", raw[:300])
        return {}