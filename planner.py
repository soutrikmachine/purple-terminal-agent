"""
Hierarchical Planner — decomposes the task into ordered sub-goals.

Called ONCE per task, after Turn 0 recon.
Input:  task description + recon snapshot + detected domains + RAG hints
Output: structured JSON plan with ordered sub-goals

The planner enables genuine inference-time scaling through DEPTH:
  - Forces reasoning about the full task before any modifying command
  - Anchors execution to observed reality (recon grounds the plan)
  - Sub-goals can be re-planned mid-execution if 3 consecutive failures occur
"""

from __future__ import annotations

import logging

from llm import complete_json

logger = logging.getLogger(__name__)

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
      "estimated_turns": 2
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
"""


async def plan(
    task_text: str,
    recon_snapshot: str,
    domains_str: str,
    rag_hints: str,
    max_turns: int = 30,
) -> dict:
    """
    Generate a structured plan for the task.
    Returns the parsed plan dict, or a minimal fallback plan on failure.
    """
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

## Constraint
Max total turns available: {max_turns - 3} (reserve 3 for self-verification).

Produce the JSON plan now.
"""
    result = await complete_json(
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=1500,
        temperature=0.2,
    )

    # Validate and normalise
    if not result or "subgoals" not in result:
        logger.warning("Planner returned invalid result, using fallback plan.")
        return _fallback_plan(task_text)

    # Ensure sub-goals have required fields
    for i, sg in enumerate(result.get("subgoals", []), 1):
        sg.setdefault("id", i)
        sg.setdefault("domain", "generic")
        sg.setdefault("verification", "check output and exit code")
        sg.setdefault("estimated_turns", 3)

    logger.info(
        "Plan generated: %d sub-goals | risks=%s",
        len(result.get("subgoals", [])),
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
             "estimated_turns": 3},
            {"id": 2, "goal": "Execute the primary task requirement",
             "domain": "generic", "verification": "exit code 0 and expected output",
             "estimated_turns": 10},
            {"id": 3, "goal": "Verify solution is correct",
             "domain": "generic", "verification": "tests pass or state matches requirement",
             "estimated_turns": 3},
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
        lines.append(
            f"  [{sg['id']}] {sg['goal']} "
            f"(domain: {sg.get('domain', '?')}, "
            f"verify: {sg.get('verification', '?')})"
        )
    return "\n".join(lines)
