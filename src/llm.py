"""
LLM client — OpenRouter via OpenAI-compatible API.

v0.4: RLM-style REPL architecture.
  complete()            → free-form text (planner internals)
  complete_json()       → structured JSON (planner + critic)
  complete_with_tools() → tool-use (bash/repl/final executor loop)
  llm_query_sync()      → synchronous sub-LLM call from inside REPL thread
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

MODEL     = os.environ.get("MODEL", "deepseek/deepseek-v4-flash")
API_KEY   = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL  = "https://openrouter.ai/api/v1"
SUB_MODEL = os.environ.get("SUB_MODEL", MODEL)  # sub-LLM for llm_query inside REPL

_client:      AsyncOpenAI | None = None
_sync_client: OpenAI | None      = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def get_sync_client() -> OpenAI:
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _sync_client


def llm_query_sync(prompt: str, max_tokens: int = 2048) -> str:
    """Synchronous sub-LLM call for use inside REPL (runs in asyncio.to_thread).
    Uses SUB_MODEL (default: V4 Flash) — cheap and fast for context inspection."""
    try:
        client = get_sync_client()
        resp = client.chat.completions.create(
            model=SUB_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt[:400_000]}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[llm_query error: {e}]"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def complete(
    system: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.4,
    model_override: str | None = None,
) -> str:
    """Free-form text completion. Used by planner internals."""
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
    """JSON-mode completion. Used by planner and critic."""
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
        logger.error("Failed to parse JSON from LLM: %s", raw[:300])
        return {}


# ── Tool definitions for RLM executor loop ───────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the task environment. "
                "Output appended to `context` list in the REPL. "
                "Full output always in context[-1]['stdout']. "
                "Use for any shell action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run."},
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds; minimum 60, maximum 300. Default 300. Use 300 for installs, builds, long scripts. Never use <60.",
                        "minimum": 60, "maximum": 300,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repl",
            "description": (
                "Execute Python in a persistent in-process REPL. "
                "Globals: `context` (list of all bash/repl results), "
                "`llm_query(prompt)` (fast sub-LLM for large output processing). "
                "Use context[-1]['stdout'] to inspect full bash output. "
                "NEVER run shell commands here — use bash instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final",
            "description": "Terminate the task. Call ONLY when verified complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output": {"type": "string", "description": "Brief summary of what was accomplished."},
                },
                "required": ["output"],
            },
        },
    },
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def complete_with_tools(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.2,
    model_override: str | None = None,
) -> dict:
    """Tool-use completion for the REPL executor loop.
    Returns {name, arguments, raw_message}."""
    client = get_client()
    full_messages = [{"role": "system", "content": system}] + messages
    response = await client.chat.completions.create(
        model=model_override or MODEL,
        messages=full_messages,  # type: ignore[arg-type]
        tools=TOOLS,  # type: ignore[arg-type]
        tool_choice="required",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, ValueError):
            args = {}
        return {"name": tc.function.name, "arguments": args, "raw_message": msg}
    # No tool call despite tool_choice=required — fallback
    content = (msg.content or "").strip()
    logger.warning("No tool call despite tool_choice=required — parsing content as bash")
    return {"name": "bash", "arguments": {"command": content or "ls -la"}, "raw_message": msg}