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
# Context filter — cheap model for summarising large command outputs before main LLM
CONTEXT_FILTER_MODEL = os.environ.get("CONTEXT_FILTER_MODEL", MODEL)

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


FILTER_SYSTEM = """You are a command output summariser for a terminal agent.
You receive raw stdout/stderr from a bash command and return a concise structured summary.

Rules:
- Keep all error messages VERBATIM (they are critical for debugging)
- Keep all file paths, package names, version numbers verbatim
- Keep the last 10 lines of output verbatim (most recent state matters)
- Summarise repetitive lines (e.g. "Downloading... 1%... 2%...") into one line
- Total output MUST be under 60 lines

Format:
SUMMARY: <one sentence of what happened>
KEY_INFO: <important values: paths, versions, package names, ports>
ERRORS: <any error messages verbatim, or "none">
LAST_LINES:
<last 10 lines of original output verbatim>"""


async def summarize_output(
    stdout: str,
    stderr: str,
    command: str,
    exit_code: int,
    char_threshold: int = 3000,
) -> str:
    """
    If combined output exceeds char_threshold, use CONTEXT_FILTER_MODEL to summarise.
    Returns filtered summary string, or original output if under threshold.
    Falls back to simple truncation on any error — never blocks execution.
    """
    combined = stdout + stderr
    if len(combined) <= char_threshold:
        return combined  # Under threshold — no filtering needed

    logger.info("Context filter: output len=%d exceeds %d — summarising with %s",
                len(combined), char_threshold, CONTEXT_FILTER_MODEL)
    try:
        client = get_client()
        user_msg = (
            f"Command: `{command}` (exit_code={exit_code})\n\n"
            f"stdout (len={len(stdout)}):\n{stdout[:8000]}\n\n"
            f"stderr (len={len(stderr)}):\n{stderr[:2000]}"
        )
        response = await client.chat.completions.create(
            model=CONTEXT_FILTER_MODEL,
            messages=[
                {"role": "system", "content": FILTER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=600,
            temperature=0.0,
        )
        summary = response.choices[0].message.content or ""
        logger.info("Context filter: reduced %d → %d chars", len(combined), len(summary))
        return f"[FILTERED by context filter — original {len(combined)} chars]\n{summary}"
    except Exception as e:
        logger.warning("Context filter failed (%s) — using truncation fallback", e)
        # Simple truncation fallback
        head = combined[:1500]
        tail = combined[-1500:]
        return f"{head}\n\n...[{len(combined)-3000} chars truncated]...\n\n{tail}"