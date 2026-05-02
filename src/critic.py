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

# ── 30-second timeout patterns ───────────────────────────────────────────────
# Commands that reliably exceed the 30s hard timeout and must be rewritten.
# NOTE: "timeout N apt-get update" and "apt-get update 2>&1 | tail -N" are ALLOWED
# because they are explicitly time-bounded or output-bounded.
_TIMEOUT_PATTERNS = [
    # bare "apt-get update" with no timeout wrapper or pipe — takes 15-20s
    (r"^\s*apt-get\s+update\s*$",
     "apt-get update alone takes 15-20s — skip it or use: timeout 20 apt-get update 2>&1 | tail -3"),
    # combined apt-get update && apt-get install always exceeds 30s
    (r"apt-get\s+update\s*(?:2>&1[^&]*)?\s*&&\s*apt-get\s+install",
     "apt-get update && apt-get install always times out — drop the update, use: apt-get install -y PACKAGE 2>&1 | tail -5"),
    (r"apt\s+update\s*(?:2>&1[^&]*)?\s*&&\s*apt\s+install",
     "apt update && apt install always times out — drop the update"),
    # Large ML packages that exceed 30s download/install window
    (r"pip\s+install.*(torch|pytorch|tensorflow|easyocr|nvidia-|transformers)", 
     "Large ML packages will exceed the 30s timeout. Skip or use a smaller alternative."),
    # Aggressive security scans
    (r"nmap\s+-[as][S|V|C]", 
     "Aggressive nmap scans often exceed 30s. Use fast/targeted scans like nmap -p- or nmap -F."),
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
Your primary goal is to REPAIR commands so they survive a strict 30-second timeout and do not hang.

## Mandatory Heuristics (in priority order):

1. TIMEOUT PREVENTION (30s Budget):
   - `apt-get update` is a turn-waster. Silently REMOVE it from chained commands.[cite: 16]
   - If a command is just `apt-get update`, REVISE to `echo "Skipping update"`.[cite: 16]
   - Combined `update && install` MUST be split; remove the update and only keep the install.[cite: 16]

2. INSTALLATION STRATEGY:
   - FRAGMENTATION: If installing multiple large packages (e.g., pandas+pgmpy, torch, tensorflow), REVISE to install only ONE package per turn.[cite: 16]
   - CHECK BEFORE INSTALL: If installing a Python package, REVISE to: `python3 -c "import X" 2>/dev/null || pip install --break-system-packages X 2>&1 | tail -n 5`.[cite: 16, 19]
   - OUTPUT BOUNDING: Every `apt` or `pip` command MUST end with `2>&1 | tail -n 5` to prevent terminal buffer hangs.[cite: 16]

3. INTERACTIVE HANGS:
   - REVISE any command that opens an editor/pager (vim, nano, less, man).[cite: 16]
   - Force `GIT_SEQUENCE_EDITOR=":"` for rebases and `--no-edit` for commits/merges.[cite: 16]

4. GROUNDING & BLIND COPYING:
   - REVISE if the command references a file path not yet seen in the observation history.[cite: 16]
   - Stop the agent from assuming directory structures (e.g., assuming a `tests/` folder exists before `ls` confirms it).[cite: 16]

5. NON-DESTRUCTIVE MODS:
   - REVISE `rm` or overwrite commands on files that haven't been `cat`'d or inspected first.[cite: 16]

6. WRONG FLAGS FOR CONTEXT:
   - --force / --hard when task says "preserve history" → REVISE
   - Missing -y on apt/pip when running non-interactively → REVISE
   - Missing -L on curl for redirect-following downloads → REVISE
   - `apt-get update && apt-get install` in one command → REVISE (combined command will timeout in 30s; drop the update, just run `apt-get install -y PACKAGE 2>&1 | tail -5`)
   - `apt-get update` alone → REVISE (takes 15-20s alone, drop it entirely)
   - IMPORTANT: `pip install --break-system-packages` IS CORRECT and safe in Docker containers. Do NOT revise this flag away.

## Output Format (JSON only)
{
  "verdict": "APPROVE" or "REVISE",
  "issue": "brief description of the failure mode",
  "revised_command": "the corrected command to be executed immediately"
}

Lean toward REVISE for any multi-package install or update command.[cite: 16]
"""


def _fast_check(command: str) -> tuple[bool, str, str]:
    """
    Local pattern-based pre-check before LLM critic call.
    Returns (needs_revision, issue, suggested_fix).
    """
    for pattern, issue in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command):
            return True, issue, ""

    # 30-second timeout patterns — highest priority
    for pattern, issue in _TIMEOUT_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            # Auto-fix: strip "apt-get update &&" from combined commands
            fixed = re.sub(r"(apt-get|apt)\s+update\s*(-qq|-q)?\s*(2>&1\s*\|\s*tail\s+-\d+\s*)?&&\s*", "", command)
            
            # If it was just "apt-get update" alone, replace with a no-op explanation
            if re.match(r"^\s*(apt-get|apt)\s+update\s*$", command.strip(), re.IGNORECASE):
                fixed = "echo 'Skipping update (30s limit). Proceeding to direct install.'"
            
            # For large pip installs or nmap, no simple auto-fix; return empty string to force agent to re-plan
            if any(kw in command for kw in ["pip install", "nmap"]):
                 return True, issue, ""

            return True, issue, fixed.strip()

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