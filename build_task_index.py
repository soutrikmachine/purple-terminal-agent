"""
Build-time task indexer for Terminal Bench RAG.

Run at Docker build time to clone the public terminal-bench-2 repo
and extract task data into a structured JSON index at /app/data/task_index.json.

What is extracted per task:
  - name, domain, instruction (summary)
  - key_tools: tools seen in solve.sh (without full solution — anti-satiation)
  - gotchas: known tricky patterns detected heuristically

Crucially: we do NOT store full oracle solutions. Only tool names and
diagnostic patterns are extracted to guide REASONING, not enable copying.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


# ── Tool/pattern extraction ────────────────────────────────

# Tools to detect in solve.sh (what was used, not what to do)
_TOOL_PATTERNS = {
    "git":        r"\bgit\b",
    "docker":     r"\bdocker\b",
    "python3":    r"\bpython3?\b",
    "pip":        r"\bpip3?\b",
    "make":       r"\bmake\b",
    "cmake":      r"\bcmake\b",
    "curl":       r"\bcurl\b",
    "wget":       r"\bwget\b",
    "sed":        r"\bsed\b",
    "awk":        r"\bawk\b",
    "jq":         r"\bjq\b",
    "sqlite3":    r"\bsqlite3\b",
    "psql":       r"\bpsql\b",
    "systemctl":  r"\bsystemctl\b",
    "crontab":    r"\bcrontab\b",
    "chmod":      r"\bchmod\b",
    "find":       r"\bfind\b",
    "grep":       r"\bgrep\b",
    "ssh":        r"\bssh\b",
    "tar":        r"\btar\b",
    "cargo":      r"\bcargo\b",
    "npm":        r"\bnpm\b",
}

# Heuristic gotcha detection from solve.sh content
_GOTCHA_PATTERNS = [
    (r"GIT_SEQUENCE_EDITOR",    "Non-interactive git rebase requires GIT_SEQUENCE_EDITOR"),
    (r"--no-edit",              "git merge/commit --no-edit prevents editor opening"),
    (r"-y\b",                   "Package manager needs -y flag for non-interactive mode"),
    (r"DEBIAN_FRONTEND=noninteractive", "apt-get needs DEBIAN_FRONTEND=noninteractive"),
    (r"curl.*-L\b",             "curl -L needed to follow redirects"),
    (r"chmod \+x",              "Script needs executable permission before running"),
    (r"source.*activate",       "Virtual environment must be activated before use"),
    (r"pg_isready",             "Check postgres is ready before connecting"),
    (r"dos2unix",               "File may have Windows line endings"),
    (r"2>/dev/null",            "Some commands produce stderr that should be suppressed"),
]


def _extract_domain(instruction: str, solve_content: str) -> str:
    """Simple heuristic domain detection from instruction + solve.sh."""
    text = (instruction + " " + solve_content).lower()
    if re.search(r"\bgit\b", text): return "git"
    if re.search(r"\bdocker\b", text): return "docker"
    if re.search(r"\bpython\b|\bpip\b", text): return "python"
    if re.search(r"\bsql\b|\bsqlite\b|\bpostgres\b", text): return "database"
    if re.search(r"\bcurl\b|\bwget\b|\bhttp\b", text): return "network"
    if re.search(r"\bmake\b|\bcmake\b|\bgcc\b", text): return "build"
    if re.search(r"\bcron\b|\bsystemctl\b|\bchmod\b", text): return "system"
    if re.search(r"\bjq\b|\bsed\b|\bawk\b|\bcsv\b|\bjson\b", text): return "text"
    return "generic"


def _extract_key_tools(solve_content: str) -> list[str]:
    """Extract tool names used in solve.sh — not commands, just tool names."""
    tools = []
    for tool, pattern in _TOOL_PATTERNS.items():
        if re.search(pattern, solve_content):
            tools.append(tool)
    return tools


def _extract_gotchas(solve_content: str) -> list[str]:
    """Detect known tricky patterns in solve.sh."""
    gotchas = []
    for pattern, description in _GOTCHA_PATTERNS:
        if re.search(pattern, solve_content):
            gotchas.append(description)
    return gotchas


def _summarize_instruction(instruction: str) -> str:
    """Extract first meaningful sentence as summary (max 200 chars)."""
    lines = [l.strip() for l in instruction.split("\n") if l.strip()]
    if not lines:
        return ""
    # First non-empty, non-header line
    for line in lines:
        if not line.startswith("#") and len(line) > 20:
            return line[:200]
    return lines[0][:200]


def _process_task_dir(task_dir: Path) -> dict | None:
    """Process a single task directory. Returns task dict or None if invalid."""
    # Look for instruction file
    instruction_file = None
    for name in ["instruction.md", "task.md", "README.md", "task.txt"]:
        f = task_dir / name
        if f.exists():
            instruction_file = f
            break
    if not instruction_file:
        return None

    instruction = instruction_file.read_text(errors="replace")

    # Look for solve script
    solve_content = ""
    for name in ["solve.sh", "solution.sh", "oracle.sh"]:
        f = task_dir / name
        if f.exists():
            solve_content = f.read_text(errors="replace")
            break

    if not instruction and not solve_content:
        return None

    domain    = _extract_domain(instruction, solve_content)
    key_tools = _extract_key_tools(solve_content)
    gotchas   = _extract_gotchas(solve_content)
    summary   = _summarize_instruction(instruction)

    return {
        "name":        task_dir.name,
        "domain":      domain,
        "instruction": instruction[:500],   # capped — we don't want full text in index
        "summary":     summary,
        "key_tools":   key_tools,
        "gotchas":     gotchas,
        "tags":        f"{domain} {' '.join(key_tools)}",
    }


def build_index(repo_url: str, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Cloning {repo_url} ...", flush=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", repo_url, tmpdir],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            print(f"ERROR: git clone failed: {e.stderr.decode()}", file=sys.stderr)
            print("Writing empty index — RAG will be disabled.", file=sys.stderr)
            output.write_text(json.dumps({"tasks": [], "count": 0}, indent=2))
            return
        except subprocess.TimeoutExpired:
            print("ERROR: git clone timed out.", file=sys.stderr)
            output.write_text(json.dumps({"tasks": [], "count": 0}, indent=2))
            return

        repo = Path(tmpdir)

        # Find task directories — look for common structures
        tasks = []
        search_roots = [
            repo / "tasks",
            repo / "task",
            repo / "terminal_bench" / "tasks",
            repo,
        ]

        for root in search_roots:
            if not root.exists():
                continue
            for task_dir in sorted(root.iterdir()):
                if not task_dir.is_dir():
                    continue
                if task_dir.name.startswith("."):
                    continue
                task = _process_task_dir(task_dir)
                if task:
                    tasks.append(task)
            if tasks:
                break

        if not tasks:
            # Fallback: search whole repo for instruction.md files
            for instruction_file in repo.rglob("instruction.md"):
                task = _process_task_dir(instruction_file.parent)
                if task:
                    tasks.append(task)

        index = {"tasks": tasks, "count": len(tasks)}
        output.write_text(json.dumps(index, indent=2))
        print(f"Index written: {len(tasks)} tasks → {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="https://github.com/laude-institute/terminal-bench-2",
        help="Public git repo URL to clone for task extraction",
    )
    parser.add_argument(
        "--output",
        default="/app/data/task_index.json",
        help="Output path for the task index JSON",
    )
    args = parser.parse_args()
    build_index(args.repo, args.output)
