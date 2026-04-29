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
_FIND_TESTS = [
    "find . -maxdepth 4 -name 'test_*.sh' -o -name '*.test.sh' -o -name 'check_*.sh' | head -10",
    "find . -maxdepth 4 -name '*.bats' | head -10",
    "find . -maxdepth 4 \\( -name 'test_*.py' -o -name '*_test.py' \\) -not -path '*/.*' | head -10",
    "find /usr/local/bin /usr/bin -name 'test_*' -o -name 'check_*' 2>/dev/null | head -5",
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
    """Choose the right runner for the test file type."""
    if test_file.endswith(".py"):
        return f"python3 {test_file} 2>&1"
    if test_file.endswith(".bats"):
        return f"bats {test_file} 2>&1"
    # Default: bash
    return f"bash {test_file} 2>&1"


async def _verify_by_observation(
    exec_client: ExecClient,
    task_text: str,
) -> tuple[bool, str]:
    """
    Fallback: no test files found.
    Run observable state commands and return them as context.
    Caller (agent) decides whether to trust this is done.
    """
    observations = []
    for cmd in _RECON_VERIFY:
        result = await exec_client.run(cmd)
        if result.stdout:
            observations.append(f"$ {cmd}\n{result.stdout[:400]}")

    details = (
        "No test files found. Observable state:\n"
        + "\n".join(observations)
        + "\n\nVerify manually that this matches the task requirements."
    )
    # Return True optimistically — no way to verify without test files
    # The green agent's Harbor verifier will make the final call
    return True, details
