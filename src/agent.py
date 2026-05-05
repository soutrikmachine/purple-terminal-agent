"""
Purple Terminal Agent — RLM-style REPL architecture.

Three tools per turn: bash / repl / final
  bash  → sends exec_request to green, waits for exec_result
  repl  → executes Python in-process (no A2A round trip), accesses persistent context list
  final → ends the task

Persistent context list:
  self.transcript is a Python list appended after every bash and repl call.
  It persists across all A2A turns because AgentSession lives in memory.
  The LLM can inspect any past output via: context[-1]['stdout'], context[2]['command'], etc.
  This replaces simulated shell persistence (cwd tracking) with Python object persistence.

llm_query inside REPL:
  The agent calls llm_query(prompt) from inside a repl block to process large outputs
  without burning bash turns or A2A round trips. Uses SUB_MODEL (V4 Flash by default).

Pipeline preserved:
  Domain detection → Specialist prompt → RAG → Planner → Critic per bash call → Memory
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import re
import traceback

from critic import preflight
from llm import (
    complete_json,
    complete_with_tools,
    llm_query_sync,
)
from memory import get_memory
from planner import format_plan_for_executor, plan
from rag import query_rag
from specialist import build_system_prompt, detect_domains

logger = logging.getLogger("agent")

MAX_TURNS              = int(os.getenv("MAX_TURNS", "30"))
MAX_INNER_STEPS        = 8   # repl steps per A2A turn before forcing bash/final
MAX_BASH               = 60  # total bash calls per task
MAX_REPL               = 60  # total repl calls per task
MAX_LLM_QUERY          = 20  # llm_query calls inside repl
CONSECUTIVE_FAIL_LIMIT = 3

# ── Exec URL extraction (kept for test_agent.py compatibility) ───────────────
_EXEC_URL_PATTERNS = [
    r"exec[_\s-]?url[:\s]+(\S+)",
    r"POST\s+(https?://\S+/exec/\S+)",
    r"(https?://[^\s]+/exec/[^\s]+)",
    r"shell[_\s-]?url[:\s]+(\S+)",
    r"exec[_\s-]?endpoint[:\s]+(\S+)",
]


def extract_exec_url(message: str) -> str | None:
    for pattern in _EXEC_URL_PATTERNS:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;")
    return None


# ── RECON command (Turn 0) ───────────────────────────────────────────────────
RECON_CMD = (
    "echo '=== PWD ===' && pwd && "
    "echo '=== LS ===' && ls -la && "
    "echo '=== FILES ===' && find . -maxdepth 2 -not -path '*/.*' -type f | sort | head -40 && "
    "echo '=== ENV ===' && env | grep -vE '(PATH|SHELL|HOME|TERM)' | sort | head -20 && "
    "echo '=== GIT ===' && git log --oneline -5 2>/dev/null || echo '(no git)' && "
    "echo '=== TOOLS ===' && which python3 pip docker git curl wget make 2>/dev/null | head -10"
)


# ── System prompt for tool-use executor ─────────────────────────────────────
REPL_SYSTEM_SUFFIX = """
## Tools
You have exactly three tools: bash, repl, final.

**bash(command, timeout=30)**: Run a shell command. Output appended to `context` list.
  - Full stdout always in context[-1]['stdout'] — inspect via repl if truncated in chat
  - timeout clamped to [1, 300]s — set appropriately for slow operations

**repl(code)**: Execute Python in-process. Globals: `context` (transcript list), `llm_query(prompt)`.
  - context[-1] = last bash result: {'kind':'bash','command':...,'stdout':...,'stderr':...,'exit_code':...}
  - Use for: inspecting large outputs, extracting specific lines, calling llm_query
  - llm_query(prompt) → fast sub-LLM for summarising huge outputs (use for outputs >5KB)
  - Example: `print(context[-1]['stdout'][:3000])`
  - Example: `result = llm_query(f"Extract key schedule from: {context[-1]['stdout'][:20000]}")`
  - NEVER run shell commands in repl — use bash tool

**final(output)**: End the task. Call ONLY after verifying the solution is correct.

## Discipline
- One tool call per response
- After each bash, check context[-1] via repl if output seems truncated
- Verify your changes work before calling final — score is based on automated tests
- For outputs >5KB: use repl with llm_query rather than re-reading in bash
- When done: call final, not another bash
"""


class AgentSession:
    """
    One session per task (contextId). Maintains state across A2A turns.
    All bash results stored in self.transcript — persistent Python object,
    never loses data between turns.
    """

    def __init__(self, instruction: str):
        self.instruction = instruction
        self.turn        = 0
        self.done        = False
        self.bash_count  = 0
        self.repl_count  = 0
        self.llm_query_count = 0
        self.consecutive_failures = 0
        self.current_sg_idx = 0
        self.subgoals: list[dict] = []
        self.collected_commands: list[str] = []

        # ── Persistent transcript (the REPL context) ──────────────────────────
        # Every bash result is appended here. Persists across all A2A turns.
        self.transcript: list[dict] = []
        self.repl_globals: dict = {}
        self._repl_initialized = False

        # ── Conversation history for tool-use LLM ────────────────────────────
        self.history: list[dict] = []

        # ── State for pending bash (waiting for exec_result) ─────────────────
        self.pending_bash_command: str | None = None
        self.pending_bash_tool_id: str | None = None

        # ── Domain detection + specialist prompt ──────────────────────────────
        self._domain_result = detect_domains(instruction)
        logger.info("Domains: primary=%s secondaries=%s",
                    self._domain_result.primary, self._domain_result.secondaries)

        rag_hints  = query_rag(instruction, top_k=2)
        memory_ctx = get_memory().format_for_injection(self._domain_result.primary)

        base_prompt = build_system_prompt(self._domain_result, rag_hints, MAX_TURNS)
        if memory_ctx:
            base_prompt += "\n\n" + memory_ctx
        self.system_prompt = base_prompt + REPL_SYSTEM_SUFFIX

    def _init_repl(self) -> None:
        if self._repl_initialized:
            return
        llm_query_count_ref = [0]
        transcript_ref = self.transcript

        def llm_query(prompt: str) -> str:
            llm_query_count_ref[0] += 1
            self.llm_query_count += 1
            if self.llm_query_count > MAX_LLM_QUERY:
                return "[llm_query budget exhausted]"
            return llm_query_sync(prompt[:400_000])

        self.repl_globals = {
            "context": transcript_ref,
            "llm_query": llm_query,
            "json": json,
            "re": re,
        }
        self._repl_initialized = True

    def _exec_repl(self, code: str) -> tuple[str, str, str | None]:
        """Execute Python code in persistent REPL. Returns (stdout, stderr, exception)."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exc: str | None = None
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, self.repl_globals)  # noqa: S102
        except Exception:
            exc = traceback.format_exc()
        return stdout_buf.getvalue(), stderr_buf.getvalue(), exc

    # ── Entry points called from server.py ───────────────────────────────────

    async def on_task(self) -> str:
        """Phase 1: send RECON command first."""
        self._init_repl()
        logger.info("Session start — sending RECON command")
        self.pending_bash_command = RECON_CMD
        return json.dumps({"kind": "exec_request", "command": RECON_CMD, "timeout": 300})

    async def on_exec_result(self, result: dict) -> str:
        """Called for every exec_result from green."""
        self.turn += 1
        stdout    = str(result.get("stdout", "") or "").strip()
        stderr    = str(result.get("stderr", "") or "").strip()
        exit_code = int(result.get("exit_code", result.get("returncode", 0)))
        timed_out = bool(result.get("timed_out", False))

        logger.info("Turn %d exit=%d timed_out=%s stdout_len=%d",
                    self.turn, exit_code, timed_out, len(stdout))

        # Append to persistent transcript
        entry = {
            "kind":     "bash",
            "command":  self.pending_bash_command or "",
            "exit_code": exit_code,
            "stdout":   stdout,
            "stderr":   stderr,
            "timed_out": timed_out,
        }
        self.transcript.append(entry)
        self.repl_globals["context"] = self.transcript

        # ── Phase 1: recon result → planning ─────────────────────────────────
        if self.turn == 1:
            recon_snapshot = f"stdout:\n{stdout}\nstderr:\n{stderr}"
            logger.info("Phase 2: Planning")
            domains_str = (
                f"Primary: {self._domain_result.primary}, "
                f"Secondary: {', '.join(self._domain_result.secondaries) or 'none'}"
            )
            try:
                plan_data = await plan(
                    task_text=self.instruction,
                    recon_snapshot=recon_snapshot,
                    domains_str=domains_str,
                    rag_hints=query_rag(self.instruction, top_k=2),
                    max_turns=MAX_TURNS - self.turn,
                )
            except Exception as e:
                logger.warning("Planner failed: %s — using fallback", e)
                from planner import _fallback_plan
                plan_data = _fallback_plan(self.instruction)

            self.subgoals = plan_data.get("subgoals", [])
            plan_summary  = format_plan_for_executor(plan_data)
            logger.info("Plan: %s", plan_summary[:300])

            # Seed conversation
            self.history = [{
                "role": "user",
                "content": (
                    f"Task:\n{self.instruction}\n\n"
                    f"Environment Recon:\n```\n{recon_snapshot[:3000]}\n```\n\n"
                    f"{plan_summary}\n\n"
                    "Begin executing the plan. Use bash for shell commands, "
                    "repl to inspect large outputs or call llm_query, "
                    "final when complete."
                ),
            }]
        else:
            # Build truncated observation for chat (full in context[-1])
            combined = stdout + stderr
            preview = combined[:6000] if len(combined) <= 6000 else (
                combined[:3000] + f"\n...[{len(combined)-6000} chars; full in context[-1]]\n" + combined[-3000:]
            )
            obs = (
                f"exit_code={exit_code}\n"
                f"{'⏱ Command timed out\n' if timed_out else ''}"
                f"stdout (preview):\n{preview[:4000]}\n"
                f"stderr: {stderr[:500] if stderr else '(none)'}\n\n"
                f"Full output always in context[-1]['stdout'] — inspect via repl if needed."
            )

            # Update subgoal tracking from LLM output
            remaining = MAX_TURNS - self.turn
            urgency = ""
            if remaining <= 3:
                urgency = f"\n🚨🚨🚨 ONLY {remaining} TURNS LEFT. Call final NOW."
            elif remaining <= 6:
                urgency = f"\n🚨 {remaining} turns left — wrap up."
            elif remaining <= 10:
                urgency = f"\n⚠️ {remaining} turns remaining."

            current_sg = (self.subgoals[min(self.current_sg_idx, len(self.subgoals)-1)]
                          if self.subgoals else {})
            sg_info = f"\nCurrent sub-goal: [{current_sg.get('id','?')}] {current_sg.get('goal','')}" if current_sg else ""

            self.history.append({
                "role": "user",
                "content": obs + urgency + sg_info,
            })

            # Track consecutive failures
            if exit_code != 0:
                self.consecutive_failures += 1
                if self.consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
                    self.history.append({
                        "role": "user",
                        "content": (
                            f"🚨 ERROR RECOVERY: {CONSECUTIVE_FAIL_LIMIT} consecutive failures. "
                            "STOP. Use repl to inspect context and understand the root cause, "
                            "then try a completely different approach."
                        ),
                    })
                    self.consecutive_failures = 0
                    self.current_sg_idx = min(self.current_sg_idx + 1, len(self.subgoals) - 1)
            else:
                self.consecutive_failures = 0

        if self.turn >= MAX_TURNS:
            logger.warning("Max turns reached")
            return json.dumps({
                "kind": "final",
                "output": f"Max turns ({MAX_TURNS}) reached. Best-effort execution completed."
            })

        return await self._inner_loop()

    async def _inner_loop(self) -> str:
        """
        Run up to MAX_INNER_STEPS tool calls before yielding back to green.
        repl calls execute in-process immediately (no A2A round trip).
        bash calls return exec_request to green and exit the loop.
        final calls end the task.
        """
        for step in range(MAX_INNER_STEPS):
            if self.bash_count >= MAX_BASH:
                return json.dumps({"kind": "final", "output": "[bash budget exhausted]"})

            try:
                result = await complete_with_tools(
                    system=self.system_prompt,
                    messages=self.history,
                    max_tokens=1024,
                    temperature=0.2,
                )
            except Exception as e:
                logger.error("LLM error at step %d: %s", step, e)
                return json.dumps({"kind": "final", "output": f"LLM error: {e}"})

            name  = result["name"]
            args  = result["arguments"]
            raw_msg = result.get("raw_message")

            # Add assistant turn to history
            if raw_msg is not None:
                try:
                    self.history.append({"role": "assistant", "content": raw_msg.model_dump()["content"]})
                except Exception:
                    self.history.append({"role": "assistant", "content": str(raw_msg)})
            else:
                self.history.append({"role": "assistant", "content": json.dumps({"tool": name, "args": args})})

            logger.info("Tool call: %s | args preview: %s", name, str(args)[:150])

            # ── final ────────────────────────────────────────────────────────
            if name == "final":
                output = str(args.get("output", "Task completed."))
                get_memory().store(
                    domain=self._domain_result.primary,
                    task_text=self.instruction,
                    commands=self.collected_commands,
                    observations_summary=str(self.transcript[-3:])[:500],
                    verified=True,
                )
                logger.info("Task done at turn %d bash=%d repl=%d",
                            self.turn, self.bash_count, self.repl_count)
                self.done = True
                return json.dumps({"kind": "final", "output": output})

            # ── repl ─────────────────────────────────────────────────────────
            if name == "repl":
                if self.repl_count >= MAX_REPL:
                    self.history.append({
                        "role": "user",
                        "content": "[repl budget exhausted — use bash or call final]",
                    })
                    continue

                code = str(args.get("code", ""))
                self.repl_count += 1
                stdout_r, stderr_r, exc = await asyncio.to_thread(self._exec_repl, code)

                # Append to transcript
                self.transcript.append({
                    "kind": "repl", "code": code,
                    "stdout": stdout_r, "stderr": stderr_r, "exception": exc,
                })

                obs_parts = []
                if stdout_r: obs_parts.append(f"stdout:\n{stdout_r[:3000]}")
                if stderr_r: obs_parts.append(f"stderr:\n{stderr_r[:500]}")
                if exc:      obs_parts.append(f"exception:\n{exc[:500]}")
                obs = "\n\n".join(obs_parts) or "(no output)"

                self.history.append({"role": "user", "content": f"[repl result]\n{obs}"})
                logger.info("repl step %d: stdout_len=%d exc=%s", step, len(stdout_r), bool(exc))
                continue  # next inner step — still in loop

            # ── bash ─────────────────────────────────────────────────────────
            if name == "bash":
                cmd = str(args.get("command", "")).strip()
                if not cmd:
                    self.history.append({"role": "user", "content": "[empty bash command — provide a command]"})
                    continue

                tmo = int(args.get("timeout", 30))
                tmo = max(1, min(tmo, 300))

                # ── Critic pre-flight ─────────────────────────────────────────
                current_sg = (self.subgoals[min(self.current_sg_idx, len(self.subgoals)-1)]
                              if self.subgoals else {})
                obs_history = " | ".join(
                    f"{e.get('command','')[:40]}→exit={e.get('exit_code',0)}"
                    for e in self.transcript[-3:]
                    if e.get("kind") == "bash"
                )
                try:
                    final_cmd, critic_note = await preflight(
                        command=cmd,
                        subgoal=current_sg.get("goal", ""),
                        observation_history=obs_history,
                        turn_number=self.turn,
                        domain=self._domain_result.primary,
                    )
                except Exception as e:
                    logger.warning("Critic failed: %s", e)
                    final_cmd, critic_note = cmd, ""

                if critic_note:
                    logger.info("Critic revised: %s → %s | %s", cmd[:60], final_cmd[:60], critic_note)
                    self.history.append({
                        "role": "user",
                        "content": f"[critic] {critic_note} → running: `{final_cmd}`",
                    })

                self.bash_count += 1
                self.collected_commands.append(final_cmd)
                self.pending_bash_command = final_cmd

                # ── Subgoal done detection from recent history ────────────────
                last_assistant = next(
                    (m["content"] for m in reversed(self.history) if m["role"] == "assistant"),
                    ""
                )
                sg_done = re.search(r'<subgoal_done\s+id=["\']?(\d+)["\']?\s*/?>', str(last_assistant))
                if sg_done:
                    completed_id = int(sg_done.group(1))
                    new_idx = min(completed_id, len(self.subgoals) - 1)
                    if new_idx > self.current_sg_idx:
                        self.current_sg_idx = new_idx
                        logger.info("Subgoal %d complete → idx=%d", completed_id, self.current_sg_idx)

                return json.dumps({"kind": "exec_request", "command": final_cmd, "timeout": tmo})

            # Unknown tool
            logger.warning("Unknown tool: %s", name)
            self.history.append({"role": "user", "content": f"[unknown tool {name}] use bash, repl, or final"})

        # Exhausted inner steps without bash/final — nudge
        logger.warning("Inner step cap reached — forcing bash")
        self.history.append({
            "role": "user",
            "content": "Step limit reached. Call bash with your next command, or final if done.",
        })
        return json.dumps({"kind": "exec_request", "command": "echo 'step_cap_recovery' && ls -la", "timeout": 30})

    def _build_observation(self, *args, **kwargs) -> str:
        """Kept for compatibility — not used in REPL architecture."""
        return ""


class TerminalAgent:
    def __init__(self):
        self._sessions: dict[str, AgentSession] = {}

    async def handle_task(self, context_id: str, instruction: str) -> str:
        logger.info("New session context_id=%s instruction_len=%d preview=%.200s",
                    context_id[:20], len(instruction), instruction)
        session = AgentSession(instruction)
        self._sessions[context_id] = session
        return await session.on_task()

    async def handle_exec_result(self, context_id: str, payload: dict) -> str:
        session = self._sessions.get(context_id)
        if session is None:
            logger.error("No session for context_id=%s", context_id[:20])
            return json.dumps({"kind": "final", "output": "Session not found."})
        if session.done:
            return json.dumps({"kind": "final", "output": "Task already complete."})
        return await session.on_exec_result(payload)

    def handle_final(self, context_id: str) -> str:
        self._sessions.pop(context_id, None)
        return json.dumps({"kind": "final", "output": "Acknowledged."})