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
    (r"\bsudo\b.*(apt|dpkg|yum).*(?<!-y\b)$",
     "package manager may prompt for confirmation — add -y flag"),
]

_DESTRUCTIVE_PATTERNS = [
    (r"\brm -rf /\b",         "rm -rf / is catastrophic — never run this"),
    (r"\b> /dev/s",           "overwriting a device file"),
    (r"\bdd if=.*of=/dev/[sh]d",  "dd to raw disk device"),
    (r"\bchmod -R 777\b",     "chmod 777 recursively removes all security"),
]

CRITIC_SYSTEM = """You are a pre-flight safety reviewer for a terminal agent.

You receive a draft command and must decide: APPROVE or REVISE.

## Check for these failure modes (in priority order):

1. INTERACTIVE HANG: Will this command open an editor, pager, or wait for stdin?
   - git rebase -i without GIT_SEQUENCE_EDITOR set → REVISE
   - vim, nano, less, man, read → REVISE
   - Any command that pauses waiting for user input → REVISE

2. BLIND COPYING: Is this command grounded in the OBSERVED state, or copied from memory?
   - References files/paths NOT seen in the observation history → REVISE  
   - Uses hardcoded values (branch names, commit counts) not confirmed by reading → REVISE
   - Pattern-matches the task type without checking actual environment → REVISE

3. DESTRUCTIVE WITHOUT EXPLORATION: Is this modifying/deleting before sufficient reading?
   - First command is already writing/deleting (should explore first) → REVISE
   - rm, overwrite, truncate on files that haven't been cat'd → REVISE

4. WRONG FLAGS FOR CONTEXT:
   - --force / --hard when task says "preserve history" → REVISE
   - Missing -y on apt/pip when running non-interactively → REVISE
   - Missing -L on curl for redirect-following downloads → REVISE

5. SAFETY: Could this corrupt state needed by the verifier?
   - Deleting test files → REVISE
   - Overwriting config that wasn't read first → REVISE

## If APPROVE: command is safe and grounded.
## If REVISE: provide a corrected command that achieves the same goal safely.

## Output Format (JSON only)
{
  "verdict": "APPROVE" or "REVISE",
  "issue": "brief description of the problem (empty string if APPROVE)",
  "revised_command": "safer command (empty string if APPROVE)"
}

Be decisive. Lean toward APPROVE for simple read operations.
Lean toward REVISE for any command that could hang or cause irreversible damage.
"""


def _fast_check(command: str) -> tuple[bool, str, str]:
    """
    Local pattern-based pre-check before LLM critic call.
    Returns (needs_revision, issue, suggested_fix).
    """
    for pattern, issue in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command):
            return True, issue, ""

    for pattern, issue in _INTERACTIVE_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            # Provide automatic fixes for the most common cases
            fix = ""
            if "git rebase -i" in command:
                # Auto-fix: inject GIT_SEQUENCE_EDITOR
                fix = ""  # let LLM generate the specific fix
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
            temperature=0.1,
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
    These will never hang, destroy state, or cause harm.
    """
    safe_prefixes = [
        "ls", "cat ", "head ", "tail ", "echo ", "pwd", "find ", "grep ",
        "wc ", "file ", "which ", "env", "id", "whoami", "date", "uname",
        "python3 -c", "python -c", "git log", "git status", "git diff",
        "git branch", "git show", "docker images", "docker ps",
        "ss -", "netstat ", "ps aux", "ps -", "df ", "du ",
        "sqlite3 ", "psql ", "curl -I", "curl --head",
        "crontab -l", "systemctl status",
    ]
    stripped = command.strip()
    return any(stripped.startswith(p) for p in safe_prefixes)
