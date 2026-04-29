"""
Purple Terminal Agent — A2A server (FastAPI + JSON-RPC 2.0)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import TerminalAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("purple_terminal_agent")

PORT = int(os.getenv("PORT", "9009"))

logger.info("=" * 60)
logger.info("Purple Terminal Agent  port=%d", PORT)
logger.info("Model: %s", os.getenv("MODEL", "deepseek/deepseek-v4-flash"))
logger.info("OpenRouter key: %s", "SET ✓" if os.getenv("OPENROUTER_API_KEY") else "MISSING ✗")
logger.info("=" * 60)

app    = FastAPI(title="Purple Terminal Agent")
_agent: TerminalAgent | None = None


def get_agent() -> TerminalAgent:
    global _agent
    if _agent is None:
        _agent = TerminalAgent()
    return _agent


AGENT_CARD = {
    "name": "Purple Terminal Agent",
    "description": (
        "Hierarchical Planner + Critic Pre-flight + RAG terminal agent "
        "for Terminal Bench 2.0."
    ),
    "url": f"http://localhost:{PORT}/",
    "version": "0.2.0",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "skills": [{
        "id": "terminal-task",
        "name": "Terminal Task Solver",
        "description": "Solves hard realistic terminal tasks via hierarchical planning.",
        "tags": ["terminal", "bash", "coding", "system-admin", "git", "docker"],
    }],
}


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return JSONResponse(content=AGENT_CARD)


@app.get("/.well-known/agent.json")
async def agent_card_compat():
    return JSONResponse(content=AGENT_CARD)


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "purple-terminal-agent", "version": "0.2.0"}


@app.post("/")
async def handle_task(request: Request):
    body = await request.json()

    jsonrpc_id  = body.get("id", str(uuid.uuid4()))
    task_id     = str(uuid.uuid4())
    artifact_id = str(uuid.uuid4())

    logger.info("─" * 50)
    logger.info("Request id=%s method=%s", jsonrpc_id, body.get("method"))

    # ── DIAGNOSTIC: log ALL request headers ──────────────────
    logger.info("=== REQUEST HEADERS ===")
    for k, v in request.headers.items():
        logger.info("  HEADER %s: %s", k, v)

    # ── DIAGNOSTIC: query AMBER_DYNAMIC_CAPS_API_URL ─────────
    amber_caps_url = os.environ.get("AMBER_DYNAMIC_CAPS_API_URL", "")
    if amber_caps_url:
        import urllib.request
        for path in ["", "/", "/caps", "/capabilities", "/exec", "/slots"]:
            try:
                url = amber_caps_url.rstrip("/") + path
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    body_bytes = r.read()
                    logger.info("=== AMBER CAPS API %s → %d ===", url, r.status)
                    logger.info("  RESPONSE: %s", body_bytes.decode()[:1000])
            except Exception as e:
                logger.info("=== AMBER CAPS API %s → ERROR: %s ===", path, e)



    # ── DIAGNOSTIC: log FULL inner JSON untruncated ───────────
    try:
        parts = body.get("params", {}).get("message", {}).get("parts", [])
        for i, part in enumerate(parts):
            text = part.get("text", "")
            logger.info("=== FULL PART[%d] (len=%d) ===", i, len(text))
            for j in range(0, len(text), 500):
                logger.info("  [%d:%d] %s", j, j+500, text[j:j+500])
    except Exception as e:
        logger.error("Diagnostic error: %s", e)

    task_text, context_id = _extract_task_and_context(body)
    if not context_id:
        context_id = str(uuid.uuid4())

    full_message = json.dumps(body)

    logger.info("context_id=%s  task_len=%d  preview=%.200s",
                context_id[:20], len(task_text), task_text)

    result_text = await get_agent().solve(full_message)

    logger.info("Task completed: len=%d preview=%.200s", len(result_text), result_text)

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": jsonrpc_id,
        "result": {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "completed"},
            "artifacts": [{
                "artifactId": artifact_id,
                "name": "result",
                "parts": [{"kind": "text", "text": result_text}],
            }],
        },
    })


def _extract_task_and_context(body: dict) -> tuple[str, str]:
    context_id = ""
    if "task" in body and isinstance(body["task"], str):
        return body["task"], context_id
    try:
        params     = body.get("params", {})
        message    = params.get("message", {})
        context_id = message.get("contextId", "") or params.get("contextId", "")
        parts = message.get("parts", [])
        logger.info("Parts: %d  contextId=%s", len(parts), context_id[:20] if context_id else "none")
        for i, part in enumerate(parts):
            kind = part.get("kind") or part.get("type", "")
            text = part.get("text", "")
            logger.info("Part[%d] kind=%s len=%d preview=%.150s", i, kind, len(text), text)
            if kind == "text" and text.strip():
                return text.strip(), context_id
            if kind == "data":
                data = part.get("data", {})
                if isinstance(data, dict):
                    return json.dumps(data), context_id
    except Exception as e:
        logger.error("Extraction error: %s", e)
    found = _deep_find(body, "text")
    if found:
        return found, context_id
    logger.warning("No task text found. Body keys: %s", list(body.keys()))
    return json.dumps(body), context_id


def _deep_find(obj: Any, key: str, depth: int = 0) -> str:
    if depth > 6:
        return ""
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], str) and obj[key].strip():
            return obj[key]
        for v in obj.values():
            r = _deep_find(v, key, depth + 1)
            if r:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _deep_find(item, key, depth + 1)
            if r:
                return r
    return ""


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")