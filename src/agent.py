"""
Main orchestrator for the Purple Terminal Agent.

Full pipeline per task:
  Phase 0 — Parse exec_url, detect domains, load memory+RAG context
  Phase 1 — Recon: fixed env fingerprint (Turn 0)
  Phase 2 — Plan: hierarchical planner produces ordered sub-goals
  Phase 3 — Execute: per sub-goal ReAct loop with critic pre-flight
             + error recovery (3 consecutive failures → sub-goal replan)
  Phase 4 — Self-verify: run test files before declaring done
  Phase 5 — Memory: store verified sequence for future tasks

Mathematical workflow:
  Given task T, exec E(cmd)→(stdout,stderr,exit):
    obs₀ = E(recon_bundle)
    plan = Planner(T, obs₀, domains, RAG)        # inference-time depth scaling
    for gᵢ in plan.subgoals:
      while not done(gᵢ) and turns_left > 0:
        cmd_draft = LLM_executor(gᵢ, history, ICL)
        cmd_final = Critic(cmd_draft, gᵢ, history) # pre-flight depth scaling
        obsₜ = E(cmd_final)
        if consecutive_failures ≥ 3: replan(gᵢ)
    passed, _ = SelfVerify(E, T)
    if passed: done()
    else: resume loop
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

from critic import preflight
from executor import ExecClient, ExecResult
from llm import complete
from memory import get_memory
from planner import format_plan_for_executor, plan
from rag import query_rag
from specialist import DomainResult, build_system_prompt, detect_domains
from verifier import self_verify


logger = logging.getLogger(__name__)

MAX_TURNS              = int(os.environ.get("MAX_TURNS", "30"))
MAX_OUTPUT_CHARS       = 6000
CONSECUTIVE_FAIL_LIMIT = 3   # replan sub-goal after this many failures
VERIFY_RETRY_LIMIT     = 2   # max times to re-enter executor after verify fails

# ── Exec URL extraction ────────────────────────────────────

_EXEC_URL_KEYS = [
    "exec_url", "execUrl", "shell_url", "shellUrl",
    "exec_endpoint", "execEndpoint", "shell_endpoint",
    "base_url", "baseUrl",
]

_EXEC_URL_PATTERNS = [
    r"exec[_\s-]?url[:\s]+(\S+)",
    r"POST\s+(https?://\S+/exec/\S+)",
    r"(https?://[^\s]+/exec/[^\s]+)",
    r"shell[_\s-]?url[:\s]+(\S+)",
    r"exec[_\s-]?endpoint[:\s]+(\S+)",
]


def _find_exec_url_in_dict(d: dict) -> str | None:
    for key in _EXEC_URL_KEYS:
        val = d.get(key)
        if val and isinstance(val, str) and val.startswith("http"):
            return val.rstrip(".,;")
    return None


def extract_exec_url(message: str) -> str | None:
    """
    Find exec_url in a (possibly double-encoded) JSON message.

    Terminal-bench sends a JSON-RPC envelope where:
      params.message.parts[0].text = JSON string containing:
        {"kind": "task", "exec_url": "http://...", "instruction": "..."}

    We must parse outer JSON → navigate to parts[0].text → parse inner JSON.
    """
    try:
        outer = json.loads(message)
        if isinstance(outer, dict):
            # Top-level check
            found = _find_exec_url_in_dict(outer)
            if found:
                return found

            # Navigate into JSON-RPC: params → message → parts → text
            parts = (outer.get("params", {})
                         .get("message", {})
                         .get("parts", []))
            for part in parts:
                text = part.get("text", "")
                if not text:
                    continue
                # Parse inner JSON (the actual task payload)
                try:
                    inner = json.loads(text)
                    if isinstance(inner, dict):
                        found = _find_exec_url_in_dict(inner)
                        if found:
                            return found
                except (json.JSONDecodeError, ValueError):
                    pass
                # Regex on raw inner text
                for pattern in _EXEC_URL_PATTERNS:
                    m = re.search(pattern, text, re.IGNORECASE)
                    if m:
                        return m.group(1).strip().rstrip(".,;")
    except (json.JSONDecodeError, ValueError):
        pass

    # Final fallback: regex on entire raw message
    for pattern in _EXEC_URL_PATTERNS:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;")
    return None


# ── Output helpers ─────────────────────────────────────────

def _truncate(text: str, n: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= n:
        return text
    h = n // 2
    return f"{text[:h]}\n\n...[{len(text)-n} chars omitted]...\n\n{text[-h:]}"


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ── Recon bundle ───────────────────────────────────────────

RECON_CMD = (
    "echo '=== PWD ===' && pwd && "
    "echo '=== LS ===' && ls -la && "
    "echo '=== FILES ===' && find . -maxdepth 2 -not -path '*/.*' -type f | sort | head -40 && "
    "echo '=== ENV ===' && env | grep -vE '(PATH|SHELL|HOME|TERM)' | sort | head -20 && "
    "echo '=== GIT ===' && git log --oneline -5 2>/dev/null || echo '(no git)' && "
    "echo '=== TOOLS ===' && which python3 pip docker git curl wget make 2>/dev/null | head -10"
)


# ── Main agent ─────────────────────────────────────────────

class TerminalAgent:

    def __init__(self):
        self._memory = get_memory()

    async def solve(self, task_message: str) -> str:
        """Entry point. Returns a summary of what was accomplished."""

        # ── Phase 0: Setup ─────────────────────────────────
        # Extract exec_url from full JSON body (JSON key lookup + regex fallback)
        exec_url = extract_exec_url(task_message)
        if not exec_url:
            amber_caps_url = os.environ.get("AMBER_DYNAMIC_CAPS_API_URL")
            if amber_caps_url:
                try:
                    # Query the /caps endpoint to find the shell capability
                    url = amber_caps_url.rstrip("/") + "/caps"
                    req = urllib.request.Request(url, headers={"Accept": "application/json"})
                    with urllib.request.urlopen(req, timeout=5) as r:
                        caps = json.loads(r.read().decode())
                        # Look for the capability typed as 'terminal-bench-shell-v1'
                        for cap in caps.get("capabilities", []):
                            if cap.get("kind") == "terminal-bench-shell-v1":
                                exec_url = cap.get("uri")
                                break
                except Exception as e:
                    logger.error("Failed to discover exec URL via Amber API: %s", e)

        if not exec_url:
            logger.error("No exec URL in message: %s", task_message[:300])
            return "ERROR: Could not find exec URL in task message."

        logger.info("exec_url=%s", exec_url)
        exec_client = ExecClient(exec_url)

        # Extract human-readable instruction for domain detection + planning.
        # Terminal-bench: outer JSON-RPC → params.message.parts[0].text → inner JSON
        # Inner JSON: {"kind":"task", "instruction":"...", "exec_url":"..."}
        instruction = task_message
        try:
            outer = json.loads(task_message)
            if isinstance(outer, dict):
                # Navigate to inner task JSON inside parts[0].text
                parts = (outer.get("params", {})
                             .get("message", {})
                             .get("parts", []))
                for part in parts:
                    text = part.get("text", "")
                    if not text:
                        continue
                    # Parse the inner JSON to get the instruction field
                    try:
                        inner = json.loads(text)
                        if isinstance(inner, dict):
                            for key in ("instruction", "task", "problem_statement", "description"):
                                if key in inner and isinstance(inner[key], str):
                                    instruction = inner[key]
                                    break
                    except (json.JSONDecodeError, ValueError):
                        # parts[0].text is plain text — use it directly
                        instruction = text
                    break
        except (json.JSONDecodeError, ValueError):
            pass

        logger.info("instruction preview: %.300s", instruction)

        domain_result = detect_domains(instruction)
        logger.info(
            "Domains detected: primary=%s secondaries=%s",
            domain_result.primary,
            domain_result.secondaries,
        )

        rag_hints   = query_rag(instruction, top_k=2)
        memory_ctx  = self._memory.format_for_injection(domain_result.primary)
        system_prompt = build_system_prompt(domain_result, rag_hints, MAX_TURNS)

        if memory_ctx:
            system_prompt = system_prompt + "\n\n" + memory_ctx

        try:
            result = await self._run_pipeline(
                task_message=instruction,
                exec_client=exec_client,
                domain_result=domain_result,
                system_prompt=system_prompt,
                rag_hints=rag_hints,
            )
        finally:
            await exec_client.close()

        return result

    async def _run_pipeline(
        self,
        task_message: str,
        exec_client: ExecClient,
        domain_result: DomainResult,
        system_prompt: str,
        rag_hints: str,
    ) -> str:

        turn = 0
        collected_commands: list[str] = []

        def turns_left() -> int:
            return MAX_TURNS - turn

        # ── Phase 1: Recon ──────────────────────────────────
        logger.info("Phase 1: Recon")
        recon_result = await exec_client.run(RECON_CMD)
        recon_snapshot = recon_result.combined
        turn += 1
        logger.debug("Recon snapshot: %s", recon_snapshot[:400])

        # ── Phase 2: Plan ───────────────────────────────────
        logger.info("Phase 2: Planning")
        domains_str = (
            f"Primary: {domain_result.primary}, "
            f"Secondary: {', '.join(domain_result.secondaries) or 'none'}"
        )
        task_plan = await plan(
            task_text=task_message,
            recon_snapshot=recon_snapshot,
            domains_str=domains_str,
            rag_hints=rag_hints,
            max_turns=turns_left(),
        )
        plan_summary = format_plan_for_executor(task_plan)
        logger.info("Plan: %s", plan_summary[:400])
        turn += 1  # planner counts as a turn (LLM call)

        # ── Phase 3: Execute per sub-goal ───────────────────
        logger.info("Phase 3: Execution (%d sub-goals)", len(task_plan["subgoals"]))

        # Shared conversation history for the executor
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Task:\n{task_message}\n\n"
                    f"Environment Recon:\n```\n{_truncate(recon_snapshot, 3000)}\n```\n\n"
                    f"{plan_summary}\n\n"
                    "Begin executing the plan from sub-goal [1]. One command at a time."
                ),
            }
        ]

        observation_history = recon_snapshot
        consecutive_failures = 0
        current_sg_idx = 0
        subgoals = task_plan["subgoals"]
        verify_retries = 0

        while turn < MAX_TURNS - 3:  # reserve 3 turns for verification
            # ── LLM executor call ───────────────────────────
            llm_response = await complete(
                system=system_prompt,
                messages=messages,
                max_tokens=1024,
                temperature=0.4,
            )
            messages.append({"role": "assistant", "content": llm_response})
            turn += 1

            # ── Check for <done> ────────────────────────────
            done_content = _extract_tag(llm_response, "done")
            if done_content is not None:
                # Agent thinks it's done — trigger self-verification first
                logger.info("Agent wants to declare done at turn %d. Running verifier.", turn)
                verified, verify_details = await self_verify(exec_client, task_message)
                turn += 2  # recon + test runs

                if verified:
                    logger.info("Self-verification PASSED.")
                    self._memory.store(
                        domain=domain_result.primary,
                        task_text=task_message,
                        commands=collected_commands,
                        observations_summary=observation_history[-500:],
                        verified=True,
                    )
                    return done_content or "Task completed and verified."

                # Verification failed — push back into loop
                verify_retries += 1
                if verify_retries >= VERIFY_RETRY_LIMIT:
                    logger.warning("Verify retry limit reached. Accepting current state.")
                    return f"Task attempted. Verification inconclusive: {verify_details[:300]}"

                logger.info("Verification FAILED. Pushing agent back. Details: %s", verify_details[:300])
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠️ Self-verification FAILED. Do NOT declare done yet.\n\n"
                        f"Verification output:\n```\n{verify_details[:1500]}\n```\n\n"
                        "Analyse the failure, identify what is still wrong, "
                        "and continue executing commands to fix it."
                    ),
                })
                consecutive_failures = 0
                continue

            # ── Extract command ─────────────────────────────
            command = _extract_tag(llm_response, "command")
            if not command:
                logger.warning("No <command> tag at turn %d. Nudging.", turn)
                messages.append({
                    "role": "user",
                    "content": (
                        "Use the required format:\n"
                        "<thought>reasoning</thought>\n"
                        "<command>bash_command</command>\n"
                        "Or <done>summary</done> if the task is fully complete."
                    ),
                })
                continue

            # ── Critic pre-flight ───────────────────────────
            current_sg = subgoals[min(current_sg_idx, len(subgoals) - 1)]
            final_cmd, critic_note = await preflight(
                command=command,
                subgoal=current_sg.get("goal", ""),
                observation_history=observation_history,
                turn_number=turn,
            )

            # ── Execute ─────────────────────────────────────
            logger.info("Turn %d executing: %s", turn, final_cmd[:150])
            exec_result: ExecResult = await exec_client.run(final_cmd)
            collected_commands.append(final_cmd)
            observation_history += f"\n$ {final_cmd}\n{exec_result.combined[:600]}"

            # ── Build observation message ────────────────────
            obs_msg = _build_observation(
                original_cmd=command,
                final_cmd=final_cmd,
                critic_note=critic_note,
                result=exec_result,
                turn=turn,
                turns_left=turns_left(),
                plan_summary=plan_summary,
                current_sg=current_sg,
            )
            messages.append({"role": "user", "content": obs_msg})

            # ── Error recovery ──────────────────────────────
            if exec_result.success:
                consecutive_failures = 0
                # Advance sub-goal index heuristically based on turn progression
                # (the LLM narrates which sub-goal it's on — we trust it)
            else:
                consecutive_failures += 1
                logger.info(
                    "Failure %d/%d for sub-goal [%d]: %s",
                    consecutive_failures,
                    CONSECUTIVE_FAIL_LIMIT,
                    current_sg.get("id", "?"),
                    exec_result.stderr[:100],
                )

                if consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
                    logger.warning(
                        "3 consecutive failures on sub-goal [%s]. Triggering replan.",
                        current_sg.get("id", "?"),
                    )
                    replan_msg = await _replan_subgoal(
                        subgoal=current_sg,
                        recent_history=observation_history[-2000:],
                        task_text=task_message,
                    )
                    messages.append({"role": "user", "content": replan_msg})
                    consecutive_failures = 0
                    # Advance to next sub-goal
                    current_sg_idx = min(current_sg_idx + 1, len(subgoals) - 1)

        # Max turns reached
        logger.warning("Max turns (%d) reached without completion.", MAX_TURNS)
        return (
            f"Agent reached max turns ({MAX_TURNS}). "
            "Final state reached via best-effort execution."
        )


def _build_observation(
    original_cmd: str,
    final_cmd: str,
    critic_note: str,
    result: ExecResult,
    turn: int,
    turns_left: int,
    plan_summary: str,
    current_sg: dict,
) -> str:
    """Build the observation message appended after each command execution."""
    lines = []

    if critic_note:
        lines.append(f"🔧 Critic revised command: {critic_note}")
        lines.append(f"   Original: `{original_cmd}`")
        lines.append(f"   Executed: `{final_cmd}`")
    else:
        lines.append(f"Executed: `{final_cmd}`")

    lines.append(f"\nOutput:\n```\n{_truncate(result.combined)}\n```")
    lines.append(f"\nExit code: {result.exit_code}")

    if result.timed_out:
        lines.append("⏱️  Command timed out — it may have hung waiting for input.")

    if not result.success:
        lines.append("⚠️  Non-zero exit. Read the error carefully before proceeding.")
        lines.append("   Ask: Is this a missing dep? Wrong path? Wrong flag? Wrong order?")

    lines.append(f"\nTurns remaining: {turns_left}")
    lines.append(f"Current sub-goal: [{current_sg.get('id','?')}] {current_sg.get('goal','')}")
    lines.append(
        "\nWhat is your next action? "
        "If this sub-goal is complete, move to the next one per the plan."
    )
    return "\n".join(lines)


async def _replan_subgoal(
    subgoal: dict,
    recent_history: str,
    task_text: str,
) -> str:
    """
    Generate a replan message after 3 consecutive failures on a sub-goal.
    This is a prompt injection — no extra LLM call, just a strong nudge.
    """
    goal = subgoal.get("goal", "current sub-goal")
    return (
        f"🚨 ERROR RECOVERY: You have failed sub-goal [{subgoal.get('id','?')}] "
        f"({goal}) 3 times in a row.\n\n"
        "STOP. Do not repeat the same approach.\n\n"
        "Required actions:\n"
        "1. Re-read the task description carefully — what exactly is required?\n"
        "2. Look at the error output above — what is the ROOT CAUSE?\n"
        "3. Consider: is there a completely different approach?\n"
        "4. Try the simplest possible diagnostic command first.\n\n"
        f"Recent failure context:\n```\n{recent_history[-800:]}\n```\n\n"
        "Start fresh with a diagnostic command."
    )