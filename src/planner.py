"""
Hierarchical Planner — decomposes the task into ordered sub-goals.

Called ONCE per task, after Turn 0 recon.
Input:  task description + recon snapshot + detected domains + RAG hints
Output: structured JSON plan with ordered sub-goals

The planner enables genuine inference-time scaling through DEPTH:
  - Forces reasoning about the full task before any modifying command
  - Anchors execution to observed reality (recon grounds the plan)
  - Sub-goals can be re-planned mid-execution if 3 consecutive failures occur

v0.3 additions:
  - Turn allocation forcing: each subgoal declares max_turns + timeout_risk
  - Failure-conditioned prompting: negative examples from real run failures
  - Best-of-N sampling: generate N plans, score and pick best (n=3)
  - PLANNER_MODEL env var: route to R1 or any other model independently
"""

from __future__ import annotations

import asyncio
import logging
import os

from llm import complete_json

logger = logging.getLogger(__name__)

# ── Model routing ─────────────────────────────────────────────────────────────
# Set PLANNER_MODEL=deepseek/deepseek-r1 to use R1 for planning only.
# Falls back to the global MODEL env var (deepseek/deepseek-v4-flash) if not set.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", os.getenv("MODEL", "deepseek/deepseek-v4-flash"))
API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ── Best-of-N config ──────────────────────────────────────────────────────────
PLAN_BEST_OF_N = int(os.getenv("PLAN_BEST_OF_N", "3"))  # set to 1 to disable


# ── FAILURE-CONDITIONED PROMPTING ─────────────────────────────────────────────
# Negative examples from real Sprint 3 run failures injected directly.
# This is the RL-inspired part: no training, just conditioning on observed failures.
FAILURE_EXAMPLES = """
## ❌ Plans That FAILED in Real Runs (learn from these)

FAIL 1 — build-pmars (command timeout):
  Subgoal: "Install build dependencies"
  What happened: agent ran `apt-get update && apt-get install -y build-essential`
  Result: Command timed out after 30 seconds → entire task failed immediately
  Fix: Check what's already installed. Skip apt-get update. Install ONE package at a time without update.

FAIL 2 — bn-fit-modify (command timeout):
  Subgoal: "Set up Python environment and install pandas, pgmpy"
  What happened: agent ran `apt-get update -qq && apt-get install -y python3-venv && pip install pandas pgmpy`
  Result: Timed out at 30s → task failed
  Fix: Check `python3 -c "import pandas"` first. If missing, use `pip install --break-system-packages pandas 2>&1 | tail -3`
  Note: `pip install --break-system-packages` IS safe and correct in Docker — do NOT avoid it.

FAIL 3 — chess-best-move (A2A total timeout ~18 minutes):
  Subgoal: "Analyse chess board image"
  What happened: agent spent 25+ turns iterating pixel-colour-counting scripts, never converging
  Result: Green's A2A client timed out the entire task
  Fix: Write ONE complete analysis script upfront. If pixel analysis is uncertain, use stockfish or
  python-chess FEN parsing instead of raw image pixel iteration.

FAIL 4 — code-from-image (stuck in install loop):
  Subgoal: "Install OCR libraries"
  What happened: tried tesseract (unavailable), then easyocr (too large, timeout risk),
  kept cycling through alternatives for 12 turns
  Fix: If tesseract not in apt, skip OCR entirely — use PIL to read text from image directly,
  or write the output based on what's visible in the task description.

## ✅ Patterns That SUCCEED

SUCCESS 1: Check before install
  `python3 -c "import X; print('OK')" 2>/dev/null || pip install --break-system-packages X 2>&1 | tail -3`

SUCCESS 2: Single complete script, not iterative pixel loops
  Write the full solution script in one cat > /app/solution.py << 'EOF' command, then run it.

SUCCESS 3: 4 subgoals max, final subgoal is always explicit verification
  [1] Explore (1-2 turns) → [2] Implement (4-6 turns) → [3] Test (2 turns) → [4] Verify (1-2 turns)
"""


# ── TURN ALLOCATION FORCING ───────────────────────────────────────────────────
# Forces the planner to reason explicitly about turn budgets per subgoal.
# The act of writing the number makes it reason about feasibility.
TURN_ALLOCATION_RULES = """
## Turn Allocation (MANDATORY — you MUST fill these fields)

For each subgoal, specify:
  "max_turns": N          — hard cap on exec_request turns for this subgoal (1-6 max per subgoal)
  "timeout_risk": true/false — does it involve apt-get install or pip install of large packages?

Allocation rules:
  - Exploration subgoal: max_turns = 2
  - Implementation subgoal: max_turns = 4-6
  - Testing subgoal: max_turns = 2
  - Verification subgoal: max_turns = 2
  - SUM of all max_turns MUST be ≤ {budget} (we reserve 3 for safety margin)
  - If timeout_risk = true: cap max_turns at 2 AND flag the risk in the "risks" array
  - If sum exceeds budget: MERGE or REMOVE subgoals until it fits
  - NEVER plan more than 5 subgoals total — too granular = runs out of turns
"""


PLANNER_SYSTEM = """You are a strategic task decomposer for a terminal agent.

Given:
  - A task description
  - The current environment snapshot (from recon commands)
  - Detected domains and specialist context
  - RAG hints from similar tasks (if any)

Your job: produce a structured JSON plan that breaks the task into ORDERED sub-goals.

## Output Format (JSON only, no markdown)
{
  "understanding": "One sentence: what is the core objective?",
  "success_condition": "What exact state must exist for the verifier to pass?",
  "domains": ["primary_domain", "secondary_domain"],
  "risks": ["risk 1", "risk 2"],
  "subgoals": [
    {
      "id": 1,
      "goal": "Clear, specific, actionable sub-goal",
      "domain": "which specialist domain applies",
      "verification": "how to confirm this sub-goal is done",
      "estimated_turns": 2,
      "max_turns": 2,
      "timeout_risk": false
    }
  ]
}

## Planning Rules
- First sub-goal should ALWAYS be further exploration if state is unclear.
- Keep sub-goals small: max ~5 commands each.
- Order matters: earlier sub-goals must not depend on later ones.
- Include an explicit verification sub-goal at the end.
- Do NOT include specific commands in the plan — that is the executor's job.
- Risks should be specific to THIS task, not generic warnings.
- Total estimated_turns must be realistic (sum should be under max_turns - 5).
- NEVER plan apt-get update as a step — it alone takes 15-20s of the 30s budget.
- If a subgoal needs a Python package: plan "check then install with --break-system-packages".
- If image analysis is needed: plan "write ONE complete script", not iterative pixel loops.
"""


def _score_plan(plan: dict, budget: int) -> float:
    """
    Score a candidate plan for Best-of-N selection.
    Higher is better.
    """
    score = 0.0
    subgoals = plan.get("subgoals", [])

    # Right number of subgoals (3-5 is ideal)
    n = len(subgoals)
    if 3 <= n <= 5:
        score += 3.0
    elif n == 2 or n == 6:
        score += 1.0
    else:
        score -= 2.0  # too few or too many

    # Has explicit verification subgoal at end
    if subgoals and any(
        "verif" in sg.get("goal", "").lower() or "test" in sg.get("goal", "").lower()
        for sg in subgoals[-2:]
    ):
        score += 3.0

    # Turn budget is realistic
    total_turns = sum(sg.get("max_turns", sg.get("estimated_turns", 3)) for sg in subgoals)
    if total_turns <= budget:
        score += 2.0
    else:
        score -= 3.0  # over budget is bad

    # Flags timeout risks explicitly
    risky = [sg for sg in subgoals if sg.get("timeout_risk", False)]
    if risky:
        score += 1.0  # aware of risks

    # Penalise if apt-get update appears in risks/goals (means it's planned)
    all_text = str(plan).lower()
    if "apt-get update" in all_text or "apt update" in all_text:
        score -= 2.0

    # Rewards single-script approach for image tasks
    if "script" in all_text or "write" in all_text:
        score += 1.0

    return score


async def plan(
    task_text: str,
    recon_snapshot: str,
    domains_str: str,
    rag_hints: str,
    max_turns: int = 30,
) -> dict:
    """
    Generate a structured plan for the task.
    Uses Best-of-N sampling: generates PLAN_BEST_OF_N candidates, returns the best.
    Returns the parsed plan dict, or a minimal fallback plan on failure.
    """
    budget = max_turns - 3  # reserve 3 for safety

    user_content = f"""## Task Description
{task_text}

## Environment Snapshot (from recon)
```
{recon_snapshot[:3000]}
```

## Detected Domains
{domains_str}

## Similar Task Patterns (RAG)
{rag_hints if rag_hints else "(none found)"}

## Hard Constraint
Max total turns available: {budget} (we reserve 3 for self-verification).
The SUM of all subgoal max_turns MUST be ≤ {budget}.

{TURN_ALLOCATION_RULES.format(budget=budget)}

{FAILURE_EXAMPLES}

Produce the JSON plan now. Remember: max_turns and timeout_risk are REQUIRED fields on every subgoal.
"""

    if PLAN_BEST_OF_N <= 1:
        # Single plan — original behaviour
        return await _single_plan(user_content, task_text)

    # ── Best-of-N: generate N plans in parallel, pick best ───────────────────
    logger.info("Best-of-N planning: generating %d candidates (model=%s)", PLAN_BEST_OF_N, PLANNER_MODEL)
    tasks = [_single_plan(user_content, task_text) for _ in range(PLAN_BEST_OF_N)]
    candidates = await asyncio.gather(*tasks, return_exceptions=True)

    valid = [c for c in candidates if isinstance(c, dict) and "subgoals" in c]
    if not valid:
        logger.warning("All %d plan candidates failed — using fallback", PLAN_BEST_OF_N)
        return _fallback_plan(task_text)

    scored = [(c, _score_plan(c, budget)) for c in valid]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_plan, best_score = scored[0]

    logger.info(
        "Best-of-%d: picked plan score=%.1f from %d valid candidates | subgoals=%d",
        PLAN_BEST_OF_N, best_score, len(valid), len(best_plan.get("subgoals", []))
    )
    return best_plan


async def _single_plan(user_content: str, task_text: str) -> dict:
    """Generate one plan candidate."""
    try:
        result = await complete_json(
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1500,
            temperature=0.4,  # slightly higher for Best-of-N diversity
            model_override=PLANNER_MODEL,
        )
    except Exception as e:
        logger.warning("Planner LLM call failed: %s", e)
        return _fallback_plan(task_text)

    if not result or "subgoals" not in result:
        logger.warning("Planner returned invalid result")
        return _fallback_plan(task_text)

    # Normalise subgoal fields
    for i, sg in enumerate(result.get("subgoals", []), 1):
        sg.setdefault("id", i)
        sg.setdefault("domain", "generic")
        sg.setdefault("verification", "check output and exit code")
        sg.setdefault("estimated_turns", 3)
        sg.setdefault("max_turns", sg["estimated_turns"])
        sg.setdefault("timeout_risk", False)

    logger.info(
        "Plan candidate: %d sub-goals | turns_total=%d | risks=%s",
        len(result.get("subgoals", [])),
        sum(sg.get("max_turns", 3) for sg in result.get("subgoals", [])),
        result.get("risks", []),
    )
    return result


def _fallback_plan(task_text: str) -> dict:
    """Minimal safe fallback when planner fails."""
    return {
        "understanding": task_text[:100],
        "success_condition": "Task requirements met and verified",
        "domains": ["generic"],
        "risks": ["Unknown — planner failed, proceeding carefully"],
        "subgoals": [
            {"id": 1, "goal": "Explore environment and understand current state",
             "domain": "generic", "verification": "clear picture of files and state",
             "estimated_turns": 2, "max_turns": 2, "timeout_risk": False},
            {"id": 2, "goal": "Execute the primary task requirement",
             "domain": "generic", "verification": "exit code 0 and expected output",
             "estimated_turns": 8, "max_turns": 8, "timeout_risk": False},
            {"id": 3, "goal": "Verify solution is correct and declare done",
             "domain": "generic", "verification": "tests pass or state matches requirement",
             "estimated_turns": 2, "max_turns": 2, "timeout_risk": False},
        ],
    }


def format_plan_for_executor(plan: dict) -> str:
    """Format the plan as a human-readable string for injection into executor context."""
    lines = [
        "## Task Plan",
        f"**Objective:** {plan.get('understanding', '')}",
        f"**Success condition:** {plan.get('success_condition', '')}",
    ]
    risks = plan.get("risks", [])
    if risks:
        lines.append(f"**Risks to watch:** {'; '.join(risks)}")
    lines.append("\n**Sub-goals (execute in order):**")
    for sg in plan.get("subgoals", []):
        timeout_flag = " ⚠️ TIMEOUT RISK" if sg.get("timeout_risk") else ""
        lines.append(
            f"  [{sg['id']}] {sg['goal']} "
            f"(max {sg.get('max_turns', '?')} turns, "
            f"verify: {sg.get('verification', '?')}){timeout_flag}"
        )
    return "\n".join(lines)