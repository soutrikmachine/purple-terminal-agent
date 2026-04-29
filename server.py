"""
A2A server — receives task messages from the Terminal Bench green agent
and dispatches to the TerminalAgent pipeline.

Endpoints:
  GET  /.well-known/agent.json  → AgentCard
  GET  /health                  → health check
  POST /                        → A2A task handler
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarlette
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    Role,
    TextPart,
)
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent import TerminalAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", "9009"))

AGENT_CARD = AgentCard(
    name="Purple Terminal Agent",
    description=(
        "Hierarchical Planner + Critic Pre-flight + RAG terminal agent "
        "for Terminal Bench 2.0. Decomposes tasks into sub-goals, "
        "pre-flights every command through a domain-aware critic, "
        "and self-verifies before declaring completion."
    ),
    url=f"http://localhost:{PORT}/",
    version="0.2.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="terminal-task",
            name="Terminal Task Solver",
            description="Solves hard realistic terminal tasks via hierarchical planning and multi-turn bash execution.",
            tags=["terminal", "bash", "coding", "system-admin", "git", "docker"],
        )
    ],
    default_input_modes=["text"],
    default_output_modes=["text"],
)


class PurpleAgentExecutor(AgentExecutor):

    def __init__(self):
        self._agent = TerminalAgent()

    async def execute(self, context: RequestContext, event_queue) -> None:
        task_text = _extract_text(context.message)
        logger.info(
            "Task received (id=%s, len=%d): %.200s",
            context.task_id, len(task_text), task_text,
        )
        try:
            result = await self._agent.solve(task_text)
            await event_queue.enqueue_event(_make_message(result))
        except Exception as e:
            logger.exception("Agent execution error: %s", e)
            await event_queue.enqueue_event(
                _make_message(f"Agent error: {type(e).__name__}: {e}")
            )

    async def cancel(self, context: RequestContext, event_queue) -> None:
        logger.info("Task cancelled: %s", context.task_id)


def _extract_text(message: Message | None) -> str:
    if not message:
        return ""
    parts = getattr(message, "parts", None) or []
    texts = []
    for part in parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
        elif hasattr(part, "text"):
            texts.append(str(part.text))
        elif isinstance(part, dict):
            texts.append(part.get("text", ""))
    return "\n".join(t for t in texts if t).strip()


def _make_message(text: str) -> Message:
    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.agent,
        parts=[TextPart(text=text)],
    )


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "agent": "purple-terminal-agent", "version": "0.2.0"})


def build_app():
    executor  = PurpleAgentExecutor()
    store     = InMemoryTaskStore()
    handler   = DefaultRequestHandler(agent_executor=executor, task_store=store)
    app       = A2AStarlette(agent_card=AGENT_CARD, http_handler=handler)
    app.routes.append(Route("/health", health))
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logger.info("Starting Purple Terminal Agent on %s:%d", args.host, args.port)
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
