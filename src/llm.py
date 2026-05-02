"""
LLM client — OpenRouter via OpenAI-compatible API.

All LLM calls in the system go through this module so model/key config
is in exactly one place. Provides two call modes:

  complete()       → free-form text (for executor ReAct turns)
  complete_json()  → structured JSON (for planner + critic)
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

MODEL   = os.environ.get("MODEL", "deepseek/deepseek-v4-flash")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"

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
    max_tokens: int = 2048,
    temperature: float = 0.4,
    model_override: str | None = None,
) -> str:
    """Free-form text completion. Used by executor ReAct loop."""
    client = get_client()
    full_messages = [{"role": "system", "content": system}] + messages
    response = await client.chat.completions.create(
        model=model_override or MODEL,
        messages=full_messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = response.choices[0].message.content or ""
    logger.debug("LLM complete [%d tok]: %s", max_tokens, result[:200])
    return result.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def complete_json(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.2,
    model_override: str | None = None,
) -> dict:
    """
    JSON-mode completion. Used by planner and critic.
    Lower temperature for deterministic structured output.
    Falls back to regex extraction if model doesn't return valid JSON.
    model_override: use a different model just for this call (e.g. PLANNER_MODEL=deepseek/deepseek-r1)
    """
    client = get_client()
    full_messages = [{"role": "system", "content": system}] + messages
    response = await client.chat.completions.create(
        model=model_override or MODEL,
        messages=full_messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    logger.debug("LLM json [%d tok]: %s", max_tokens, raw[:300])

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences and retry
    clean = re.sub(r"```json\s*|\s*```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Extract first {...} block
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        logger.error("Failed to parse JSON from LLM: %s", raw[:300])
        return {}