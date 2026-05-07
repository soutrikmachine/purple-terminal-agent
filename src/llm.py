"""
LLM client — OpenRouter via OpenAI-compatible API.

v0.5: Hybrid RLM/Orchestrator Architecture
  - Root Model: Gemini 3.0 Flash (Fast, flawless JSON, massive context)
  - Sub Model: DeepSeek V4 Flash (Heavy reasoning on large text inside REPL)
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# The Orchestrator (Lightning fast, immaculate JSON, massive context)
MODEL     = os.environ.get("MODEL", "google/gemini-3-flash-preview") 
# The Heavy Lifter (Used inside the REPL)
SUB_MODEL = os.environ.get("SUB_MODEL", "deepseek/deepseek-v4-flash")

API_KEY   = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL  = "https://openrouter.ai/api/v1"

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


def llm_query_sync(prompt: str, max_tokens: int = 4096) -> str:
    """Synchronous sub-LLM call for use inside REPL (runs in asyncio.to_thread)."""
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
    temperature: float = 0.25,
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
    
    # Safe regex to prevent Markdown parsing errors in your IDE
    clean = re.sub(r"`{3}json\s*|\s*`{3}", "", raw).strip()
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
            "description": "Run a shell command. Use upto 300s for any build/install/train.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Step-by-step reasoning on WHY you are running this command and what file state it will change."
                    },
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 300}
                },
                "required": ["thought", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repl",
            "description": "Execute Python in a persistent REPL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Explain what you are trying to extract or analyze from the context."
                    },
                    "code": {"type": "string"}
                },
                "required": ["thought", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final",
            "description": "Terminate the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "CRITICAL: Explain exactly how you verified the changes on the filesystem BEFORE calling final."
                    },
                    "output": {"type": "string"}
                },
                "required": ["thought", "output"],
            },
        },
    },
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
async def complete_with_tools(
    system: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.2,
    model_override: str | None = None,
) -> dict:
    """Tool-use completion for the REPL executor loop."""
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
        raw_args = tc.function.arguments or "{}"
        
        # Universal sanitization: strip <think> tags just in case a reasoning model is used
        clean_args = re.sub(r"<think>.*?</think>", "", raw_args, flags=re.DOTALL)
        clean_args = re.sub(r"`{3}json\s*|\s*`{3}", "", clean_args).strip()
        
        try:
            args = json.loads(clean_args)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Corrupted JSON in tool args, dropping to fallback.")
            args = {}
            
        return {
            "name": tc.function.name, 
            "arguments": args, 
            "tool_call_id": tc.id,
            "raw_message": msg
        }
        
    # Fallback: If it completely ignored tools and just returned text
    clean_content = re.sub(r"<think>.*?</think>", "", msg.content or "", flags=re.DOTALL).strip()
    clean_content = re.sub(r"`{3}json\s*|\s*`{3}", "", clean_content).strip()
    
    if clean_content.startswith("{"):
        try:
            parsed = json.loads(clean_content)
            if "command" in parsed:
                return {"name": "bash", "arguments": parsed, "raw_message": msg, "tool_call_id": "fallback"}
            if "code" in parsed:
                return {"name": "repl", "arguments": parsed, "raw_message": msg, "tool_call_id": "fallback"}
        except Exception:
            pass

    logger.warning("No valid tool call parsed. Forcing diagnostic bash.")
    return {"name": "bash", "arguments": {"command": "ls -la"}, "raw_message": msg, "tool_call_id": "fallback"}