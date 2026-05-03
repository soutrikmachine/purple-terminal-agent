"""
Self-verification — runs test files before the agent declares <done>.

Strategy:
  1. Find test scripts in the container (*.sh, *.bats, *.py test files)
  2. Execute each one
  3. If any fail → return False with failure details (agent loops back)
  4. If all pass → return True (agent may declare done)

If no test files found → verify by re-reading task and checking observable state.
This catches the common failure mode: task appears done but verifier disagrees.
"""

from __future__ import annotations

import logging

from executor import ExecClient, ExecResult

logger = logging.getLogger(__name__)

# Commands to find test scripts (in priority order)
# Commands to find tests in order of "Authority"
_FIND_TESTS = [
    r"find . -maxdepth 4 -name 'test_*.sh' -o -name '*.test.sh' -o -name 'check_*.sh' | head -10",
    r"find . -maxdepth 4 -name '*.bats' | head -10",
    r"find . -maxdepth 4 \( -name 'test_*.py' -o -name '*_test.py' \) -not -path '*/.*' | head -10",
    r"find /usr/local/bin /usr/bin -name 'test_*' -o -name 'check_*' 2>/dev/null | head -5",
]

_RECON_VERIFY = [
    "ls -la",
    "git log --oneline -5 2>/dev/null || true",
]


async def self_verify(exec_client: ExecClient, task_text: str) -> tuple[bool, str]:
    """
    Run self-verification.
    Returns (passed: bool, details: str).
    """
    # Step 1: find test files
    test_files: list[str] = []
    for find_cmd in _FIND_TESTS:
        result = await exec_client.run(find_cmd)
        if result.success and result.stdout:
            files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
            test_files.extend(files)
        if test_files:
            break

    if not test_files:
        logger.info("Verifier: no test files found — running observable state check.")
        return await _verify_by_observation(exec_client, task_text)

    logger.info("Verifier: found %d test file(s): %s", len(test_files), test_files[:5])

    # Step 2: run each test file
    failures: list[str] = []
    for test_file in test_files[:5]:  # max 5 test files
        run_cmd = _build_run_command(test_file)
        result = await exec_client.run(run_cmd)
        logger.info("Test %s → exit=%d", test_file, result.exit_code)
        if not result.success:
            failures.append(
                f"FAILED: {test_file} (exit={result.exit_code})\n"
                f"Output: {result.combined[:500]}"
            )

    if failures:
        details = "Self-verification FAILED:\n" + "\n---\n".join(failures)
        logger.warning("Verifier: %d test(s) failed.", len(failures))
        return False, details

    logger.info("Verifier: all %d test(s) passed.", len(test_files))
    return True, f"All {len(test_files)} test file(s) passed."


def _build_run_command(test_file: str) -> str:
    """Choose the right runner for Terminal-Bench components"""
    # Use 295s to give the executor a 5s buffer before the hard 300s timeout
    timeout_prefix = "timeout 295s " 
    
    if "test.sh" in test_file:
        return f"{timeout_prefix} bash {test_file} 2>&1"
        
    if test_file.endswith("test_outputs.py"):
        # Explicitly use python3 as required by Terminal-Bench
        return f"{timeout_prefix} python3 {test_file} 2>&1"
        
    if test_file.endswith(".py"):
        # Try pytest first for robust discovery, fallback to python3[cite: 3]
        return f"{timeout_prefix} pytest {test_file} 2>&1 || python3 {test_file} 2>&1"
        
    return f"{timeout_prefix} bash {test_file} 2>&1"


async def _verify_by_observation(
    exec_client: ExecClient,
    task_text: str,
) -> tuple[bool, str]:
    """Fallback: No benchmark scripts found. Force manual verification[cite: 3]"""
    observations = []
    # Broaden reconnaissance for new specialist domains
    verify_cmds = ["ls -R", "ps aux | grep -v grep", "ss -tlnp", "git status -s"]
    
    for cmd in verify_cmds:
        result = await exec_client.run(cmd)
        if result.stdout:
            observations.append(f"$ {cmd}\n{result.stdout[:300]}")

    details = (
        "CRITICAL: No official Terminal-Bench verification scripts found.\n"
        "Observed State:\n" + "\n".join(observations) +
        "\n\nVERIFICATION FAILED: You must perform a manual check "
        "(e.g., cat a file, curl a port, or query a DB) to confirm your "
        "changes meet all task requirements before you can declare <done>."
    )
    # Return False to force the agent to run at least one manual check[cite: 3]
    return False, details
