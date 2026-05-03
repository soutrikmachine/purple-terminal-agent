"""
Critic Pre-flight — evaluates a draft command before execution.

Called once per command in the executor loop.
Input:  draft command + current sub-goal + recent observation history
Output: APPROVE (run as-is) or REVISE (safer alternative command)

What the critic catches:
  1. Interactive commands that hang (git rebase -i without EDITOR, vim, less, etc.)
  2. Destructive operations without sufficient prior exploration
  3. Commands that blindly copy a known pattern without grounding in observed state
  4. Wrong flags for the specific environment (e.g. --force when task says preserve)
  5. Commands with hardcoded paths/values that don't match observed reality

Design: same LLM as executor, different system prompt = different reasoning lens.
This mirrors the drug-discovery architecture: specialist sub-agent as critic,
not a separate model but a separate reasoning role.
"""

from __future__ import annotations

import logging
import re

from llm import complete_json

logger = logging.getLogger(__name__)

# Fast local pre-check for obviously problematic patterns
# These are caught without an LLM call — saves latency and cost
_INTERACTIVE_PATTERNS = [
    # GIT_SEQUENCE_EDITOR may appear before or after "git rebase -i" — check whole string
    (r"^(?!.*GIT_SEQUENCE_EDITOR)(?!.*GIT_EDITOR).*\bgit rebase -i\b",
     "git rebase -i opens $EDITOR interactively — set GIT_SEQUENCE_EDITOR first"),
    (r"\bgit commit\b(?!.*-m )(?!.*--allow-empty-message)(?!.*--no-edit)",
     "git commit without -m opens editor interactively"),
    (r"\bgit merge\b(?!.*--no-edit)(?!.*-m )",
     "git merge may open editor for merge commit message — use --no-edit"),
    (r"\b(vim|vi|nano|emacs|less|more|man)\b",
     "interactive editor/pager — will hang without a TTY"),
    (r"\bread\b.*\$",
     "shell read builtin blocks waiting for stdin"),
    (r"\b(apt-get|apt)\b(?!.*\s-y)(?!.*-y\s).*\binstall\b",
     "apt install without -y flag may prompt for confirmation"),
]

# ── Unbounded Output Patterns ───────────────────────────────────────────────
# Commands that generate massive logs and cause 504 Gateway Timeouts.
_UNBOUNDED_PATTERNS = [
    (r"^(?!.*(>|tail|head)).*\b(apt-get|apt)\s+(update|install|upgrade)\b",
     "apt commands generate massive logs. Output-bound them: > /tmp/out 2>&1; head -n 20 /tmp/out; echo '...'; tail -n 40 /tmp/out"),
    (r"^(?!.*(>|tail|head)).*\bpip\s+install\b",
     "pip install progress bars flood the context. Output-bound them: > /tmp/pip.log 2>&1; head -n 20 /tmp/pip.log; echo '...'; tail -n 40 /tmp/pip.log"),
    (r"^(?!.*(>|tail|head)).*\bmake\b",
     "make builds generate too much text. Output-bound them: > /tmp/make.log 2>&1; head -n 20 /tmp/make.log; echo '...'; tail -n 40 /tmp/make.log")
]

# ── Explicitly safe patterns — skip LLM critic entirely ──────────────────────
# These are known-good patterns that should never be blocked.
_ALWAYS_SAFE_PATTERNS = [
    r"^\s*timeout\s+\d+\s+apt-get\s+update",   # timeout-bounded apt-get update
    r"^\s*timeout\s+\d+\s+apt\s+update",         # timeout-bounded apt update
    r"apt-get\s+update\s+2>&1\s*\|",             # piped (output bounded)
    r"apt\s+update\s+2>&1\s*\|",                 # piped apt update
]

_DESTRUCTIVE_PATTERNS = [
    (r"\brm -rf /\b",         "rm -rf / is catastrophic — never run this"),
    (r"\b> /dev/s",           "overwriting a device file"),
    (r"\bdd if=.*of=/dev/[sh]d",  "dd to raw disk device"),
    (r"\bchmod -R 777\b",     "chmod 777 recursively removes all security"),
]

CRITIC_SYSTEM = """You are a pre-flight safety and efficiency reviewer for a terminal agent.
Your primary goal is to REPAIR commands so they survive a strict 300-second timeout, do not hang, and provide actionable logs.

## Mandatory Heuristics (in priority order):

1. 504 GATEWAY PREVENTION (300s Budget & 60-Line Rule):
   - You have 300 seconds per command. `apt-get update`, `pip install torch`, and `make` ARE allowed.
   - However, NEVER execute `apt`, `pip`, `make`, or `npm` without output truncation.
   - REVISE unbounded commands to use the Sandwich: `(COMMAND) > /tmp/out 2>&1; head -n 20 /tmp/out; echo "... [truncated] ..."; tail -n 40 /tmp/out`

2. INSTALLATION STRATEGY:
   - OUTPUT BOUNDING: Every pip install MUST redirect output: `pip install --break-system-packages PKG > /tmp/pip.log 2>&1; tail -5 /tmp/pip.log`. You may install multiple packages in one command.
   - IMPORTANT: `pip install --break-system-packages` is safe in Docker. Do NOT remove this flag.

3. INTERACTIVE HANGS:
   - REVISE any command that opens an editor/pager (vim, nano, less, man).
   - Force `GIT_SEQUENCE_EDITOR=":"` for rebases and `--no-edit` for commits/merges.

4. GROUNDING & BLIND COPYING:
   - REVISE if the command references a file path not yet seen in history.
   - Stop the agent from assuming directory structures (e.g., assuming `tests/` exists before `ls` confirms it).

5. NON-DESTRUCTIVE MODS:
   - REVISE `rm` or overwrite commands on files that haven't been `cat`'d or inspected first.

6. WRONG FLAGS FOR CONTEXT:
   - --force / --hard when task says "preserve history" → REVISE
   - Missing -y on apt/pip when running non-interactively → REVISE
   - Missing -L on curl for redirect-following downloads → REVISE
   - IMPORTANT: `pip install --break-system-packages` IS CORRECT and safe in Docker containers. Do NOT revise this flag away.

## Output Format (JSON only)
{
  "verdict": "APPROVE" or "REVISE",
  "issue": "brief description of the failure mode",
  "revised_command": "the corrected command to be executed immediately"
}

Lean toward REVISE for any multi-package install, update command, or long-running build without output sandwiching.
"""


def _fast_check(command: str) -> tuple[bool, str, str]:
    """
    Local pattern-based pre-check before LLM critic call.
    Returns (needs_revision, issue, suggested_fix).
    """
    for pattern, issue in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command):
            return True, issue, ""

    # 504 Gateway Prevention — flag unbounded outputs
    for pattern, issue in _UNBOUNDED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            # Do not auto-fix. Return to LLM to let the Critic generate the exact sandwich fix.
            return True, issue, ""

    # Interactive patterns (keeping existing source logic)
    for pattern, issue in _INTERACTIVE_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            fix = ""
            if "git rebase -i" in command:
                fix = ""  # Let LLM generate specific sequence editor fix
            elif "git commit" in command and "-m" not in command:
                fix = command.replace("git commit", "git commit --no-edit")
            elif "git merge" in command and "--no-edit" not in command:
                fix = command + " --no-edit"
            elif "apt" in command and "-y" not in command:
                fix = command.replace("apt-get ", "apt-get -y ").replace("apt ", "apt -y ")
            return True, issue, fix

    return False, "", ""


async def preflight(
    command: str,
    subgoal: str,
    observation_history: str,
    turn_number: int,
) -> tuple[str, str]:
    """
    Pre-flight check for a draft command.

    Returns (final_command, note) where:
      - final_command is the command to actually execute (may be revised)
      - note is an explanation if revised (empty if approved as-is)
    """
    # Fast local check first (no LLM cost)
    needs_revision, issue, auto_fix = _fast_check(command)
    if needs_revision and auto_fix:
        logger.info("CRITIC fast-fix: %s → %s | %s", command[:60], auto_fix[:60], issue)
        return auto_fix, f"[auto-fixed] {issue}"

    # Skip LLM critic for simple read-only commands to save turns
    if _is_clearly_safe(command):
        return command, ""

    # LLM critic for ambiguous cases
    try:
        user_content = f"""Sub-goal: {subgoal}

Draft command: `{command}`

Recent observations (last 3 turns):
```
{observation_history[-2000:] if observation_history else "(none yet)"}
```

Turn number: {turn_number}

Is this command safe to execute? Check all 5 failure modes."""

        result = await complete_json(
            system=CRITIC_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=256,
            temperature=0.2,
        )

        verdict = result.get("verdict", "APPROVE").upper()
        issue   = result.get("issue", "")
        revised = result.get("revised_command", "").strip()

        if verdict == "REVISE" and revised and revised != command:
            logger.info("CRITIC revised: %s → %s | %s", command[:60], revised[:60], issue)
            return revised, f"[critic] {issue}"
        else:
            logger.debug("CRITIC approved: %s", command[:60])
            return command, ""

    except Exception as e:
        logger.warning("Critic failed (%s), approving command as-is: %s", e, command[:60])
        return command, ""


def _is_clearly_safe(command: str) -> bool:
    """
    Heuristic: skip LLM critic for obviously safe read-only commands.
    """
    safe_prefixes = [
        "ls", "cat ", "head ", "tail ", "echo ", "pwd", "find ", "grep ",
        "wc ", "file ", "which ", "env", "id", "whoami", "date", "uname",
        "python3 -c", "python -c", "git log", "git status", "git diff",
        "git branch", "git show", "docker images", "docker ps",
        "ss -", "netstat ", "ps aux", "ps -", "df ", "du ",
        "sqlite3 ", "psql ", "curl -I", "curl --head",
        "crontab -l", "systemctl status",
        # ML & Security Diagnostic tools
        "nvidia-smi", "nm ", "strings ", "objdump -h", "ldd ", "binwalk ",
    ]
    stripped = command.strip()
    return any(stripped.startswith(p) for p in safe_prefixes)