"""
Session-scoped task memory.

Caches ONLY verifier-confirmed successful command sequences per domain.
Retrieved sequences are injected as additional ICL at the start of each task.

Design principle: memory stores OUTCOMES not patterns.
A sequence only enters memory after self-verification passes — this prevents
the cache itself from becoming a reward-hacking source.

Scope: in-memory for the lifetime of the process (one eval run = 89 tasks).
Not persisted to disk — no state bleeds between eval runs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_PER_DOMAIN = 10     # max cached sequences per domain
MAX_COMMANDS = 15       # max commands stored per sequence
MAX_INJECT = 3          # max sequences injected per new task


@dataclass
class CommandSequence:
    domain: str
    task_summary: str           # first 100 chars of task description
    commands: list[str]         # the actual commands that worked
    observations_summary: str   # what the output looked like (brief)
    verified: bool = False      # only True after self-verification passes


class TaskMemory:
    """In-memory per-domain cache of verified successful command sequences."""

    def __init__(self):
        # domain → list of CommandSequence (most recent first)
        self._store: dict[str, list[CommandSequence]] = defaultdict(list)

    def store(
        self,
        domain: str,
        task_text: str,
        commands: list[str],
        observations_summary: str,
        verified: bool,
    ) -> None:
        """
        Store a command sequence. Only accepted if verified=True.
        Ignores unverified sequences — we don't cache failures or uncertain outcomes.
        """
        if not verified:
            logger.debug("Memory: skipping unverified sequence for domain=%s", domain)
            return
        if not commands:
            return

        seq = CommandSequence(
            domain=domain,
            task_summary=task_text[:100],
            commands=commands[:MAX_COMMANDS],
            observations_summary=observations_summary[:300],
            verified=True,
        )
        bucket = self._store[domain]
        # Prepend (most recent = most relevant)
        bucket.insert(0, seq)
        # Cap size
        if len(bucket) > MAX_PER_DOMAIN:
            bucket.pop()

        logger.info(
            "Memory: stored %d commands for domain=%s (total in domain: %d)",
            len(seq.commands),
            domain,
            len(bucket),
        )

    def retrieve(self, domain: str, n: int = MAX_INJECT) -> list[CommandSequence]:
        """Return up to n verified sequences for the given domain, most recent first."""
        return self._store.get(domain, [])[:n]

    def format_for_injection(self, domain: str) -> str:
        """
        Format retrieved sequences as scaffold hints for injection into the prompt.
        Same anti-satiation framing: process-oriented, not copy-paste templates.
        """
        sequences = self.retrieve(domain)
        if not sequences:
            return ""

        lines = [f"## Verified Patterns From Earlier Tasks (domain: {domain})"]
        lines.append("These worked in earlier tasks. Use as a REASONING REFERENCE only.")
        lines.append("Your task is different — adapt the approach, don't copy blindly.\n")

        for i, seq in enumerate(sequences, 1):
            lines.append(f"### Pattern {i} — Task: \"{seq.task_summary}\"")
            lines.append(f"Commands used (in order):")
            for cmd in seq.commands[:8]:  # show max 8 commands
                lines.append(f"  $ {cmd}")
            if seq.observations_summary:
                lines.append(f"What success looked like: {seq.observations_summary}")
            lines.append("")

        return "\n".join(lines)

    def stats(self) -> dict[str, int]:
        return {d: len(seqs) for d, seqs in self._store.items()}


# Singleton — shared across all tasks in one eval session
_memory: TaskMemory | None = None


def get_memory() -> TaskMemory:
    global _memory
    if _memory is None:
        _memory = TaskMemory()
    return _memory
