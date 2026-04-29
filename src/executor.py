"""
Exec API client — sends shell commands to the green agent's exec endpoint.

Protocol (from green agent README):
    POST /exec/{session_token}
    Body:     {"command": "bash string"}
    Response: {"stdout": "...", "stderr": "...", "exit_code": 0}

Features:
  - Adaptive timeouts (short for read ops, long for installs/builds)
  - Retry on transient network errors (not on command failures)
  - Output truncation to stay within LLM context window
  - Multiple response schema shapes handled
"""

from __future__ import annotations

import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Commands that legitimately take a long time
_LONG_RUNNING = [
    "apt-get", "apt install", "pip install", "pip3 install",
    "npm install", "yarn install", "cargo build", "cargo install",
    "docker build", "docker pull", "docker compose",
    "make ", "cmake --build", "gcc ", "g++ ",
    "wget ", "curl -", "git clone", "git fetch",
]

DEFAULT_TIMEOUT = 60.0
LONG_TIMEOUT    = 300.0
MAX_OUTPUT_CHARS = 8000  # per result (head + tail if truncated)


def _pick_timeout(command: str) -> float:
    lower = command.lower()
    for pattern in _LONG_RUNNING:
        if pattern in lower:
            return LONG_TIMEOUT
    return DEFAULT_TIMEOUT


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return f"{text[:half]}\n\n... [{omitted} chars omitted] ...\n\n{text[-half:]}"


class ExecResult:
    __slots__ = ("stdout", "stderr", "exit_code", "combined", "timed_out")

    def __init__(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        timed_out: bool = False,
    ):
        self.stdout    = stdout.strip()
        self.stderr    = stderr.strip()
        self.exit_code = exit_code
        self.timed_out = timed_out

        parts = []
        if self.stdout:
            parts.append(_truncate(self.stdout))
        if self.stderr:
            parts.append(f"[stderr]\n{_truncate(self.stderr)}")
        if not parts:
            parts.append(f"(no output, exit code: {exit_code})")
        self.combined = "\n".join(parts)

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def __repr__(self) -> str:
        return f"ExecResult(exit={self.exit_code}, out={self.stdout[:60]!r})"


def _parse(data: dict) -> ExecResult:
    """Handle multiple possible response shapes."""
    if "stdout" in data or "stderr" in data:
        return ExecResult(
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
            exit_code=int(data.get("exit_code", data.get("returncode", 0))),
        )
    if "output" in data:
        return ExecResult(
            stdout=str(data.get("output") or ""),
            stderr="",
            exit_code=int(data.get("exit_code", 0)),
        )
    if "result" in data:
        return ExecResult(stdout=str(data["result"]), stderr="", exit_code=0)
    return ExecResult(stdout=str(data), stderr="", exit_code=0)


class ExecClient:
    """HTTP client for the green agent's exec API."""

    def __init__(self, exec_url: str):
        self.exec_url = exec_url.rstrip("/")
        self._http = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            follow_redirects=True,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    async def run(self, command: str) -> ExecResult:
        """Execute a shell command and return the result."""
        timeout = _pick_timeout(command)
        logger.info("EXEC [%.0fs]: %s", timeout, command[:150])
        try:
            resp = await self._http.post(
                self.exec_url,
                json={"command": command},
                timeout=timeout,
            )
            resp.raise_for_status()
            result = _parse(resp.json())
            logger.debug(
                "EXEC → exit=%d stdout=%r stderr=%r",
                result.exit_code,
                result.stdout[:120],
                result.stderr[:80],
            )
            return result
        except httpx.TimeoutException:
            logger.warning("Command timed out (%.0fs): %s", timeout, command[:80])
            return ExecResult("", f"Command timed out after {timeout:.0f}s", 124, timed_out=True)
        except httpx.HTTPStatusError as e:
            logger.error("Exec HTTP error %s: %s", e.response.status_code, command[:80])
            return ExecResult("", f"Exec API error: {e.response.status_code}", 1)

    async def close(self) -> None:
        await self._http.aclose()
