"""
Purple Terminal Agent — A2A server (FastAPI + JSON-RPC 2.0)

Implements terminal-bench-shell-v1 multi-turn protocol:
  kind=task       → initialise session, return first exec_request
  kind=exec_result → continue session, return next exec_request or final
  kind=final      → green is done, clean up session
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid

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
logger.info("Model:          %s", os.getenv("MODEL", "deepseek/deepseek-v4-flash"))
logger.info("OpenRouter key: %s", "SET ✓" if os.getenv("OPENROUTER_API_KEY") else "MISSING ✗")
logger.info("Protocol:       terminal-bench-shell-v1 (multi-turn A2A)")
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
    "description": "Hierarchical planner terminal agent for Terminal Bench 2.0 (terminal-bench-shell-v1).",
    "url": f"http://localhost:{PORT}/",
    "version": "0.2.0",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "skills": [{
        "id": "terminal-task",
        "name": "Terminal Task Solver",
        "description": "Solves hard realistic terminal tasks via multi-turn command execution.",
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
    return {"status": "ok", "version": "0.2.0"}


@app.post("/")
async def handle_message(request: Request):
    body = await request.json()

    jsonrpc_id  = body.get("id", str(uuid.uuid4()))
    task_id     = str(uuid.uuid4())
    artifact_id = str(uuid.uuid4())

    # ── Extract contextId and inner JSON payload ─────────────
    params     = body.get("params", {})
    message    = params.get("message", {})
    context_id = message.get("contextId", "") or params.get("contextId", "") or str(uuid.uuid4())
    parts      = message.get("parts", [])

    inner_text = ""
    for part in parts:
        t = part.get("text", "")
        if t.strip():
            inner_text = t.strip()
            break

    logger.info("─" * 50)
    logger.info("context_id=%s  inner_len=%d  preview=%.300s",
                context_id[:20], len(inner_text), inner_text)

    # ── Parse inner JSON (terminal-bench-shell-v1 payload) ───
    try:
        payload = json.loads(inner_text)
    except (json.JSONDecodeError, ValueError):
        payload = {"kind": "unknown", "instruction": inner_text}

    kind = payload.get("kind", "unknown")
    logger.info("kind=%s", kind)

    # ── Route by kind ─────────────────────────────────────────
    agent = get_agent()

    if kind == "task":
        instruction = payload.get("instruction", inner_text)
        response_text = await agent.handle_task(context_id, instruction)

    elif kind == "exec_result":
        response_text = await agent.handle_exec_result(context_id, payload)

    elif kind == "final":
        response_text = agent.handle_final(context_id)

    else:
        # Unknown kind — treat as a plain task instruction
        logger.warning("Unknown kind=%s, treating as task", kind)
        response_text = await agent.handle_task(context_id, inner_text)

    logger.info("Response kind=%s  preview=%.200s",
                json.loads(response_text).get("kind"), response_text[:200])

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
                "parts": [{"kind": "text", "text": response_text}],
            }],
        },
    })


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")