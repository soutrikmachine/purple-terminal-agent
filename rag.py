"""
Task-level RAG using TF-IDF similarity over the Terminal Bench task index.

The index (data/task_index.json) is built at Docker build time by
scripts/build_task_index.py from the public terminal-bench-2 repo.

At runtime, we find the top-k most similar tasks and inject scaffold-framed
hints (NOT full oracle solutions — avoids instruction satiation / template copying).

Design principle: hints describe WHAT TO THINK ABOUT, not WHAT TO DO.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

INDEX_PATH = Path(os.environ.get("TASK_INDEX_PATH", "/app/data/task_index.json"))

# ── Pure-Python TF-IDF ──────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer — lowercase, strip punctuation, min length 2."""
    text = text.lower()
    tokens = re.findall(r"[a-z][a-z0-9_\-]{1,}", text)
    stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "or",
                 "for", "with", "that", "this", "it", "be", "on", "at",
                 "by", "as", "are", "was", "were", "has", "have", "you",
                 "your", "can", "will", "should", "must", "from", "not"}
    return [t for t in tokens if t not in stopwords]


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens) or 1
    return {t: c / n for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class TaskRAG:
    """
    TF-IDF retrieval over indexed terminal-bench tasks.
    Gracefully degrades to empty hints if index not found.
    """

    def __init__(self):
        self._tasks: list[dict] = []
        self._tfs: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if not INDEX_PATH.exists():
            logger.warning("Task index not found at %s — RAG disabled.", INDEX_PATH)
            return
        try:
            with open(INDEX_PATH) as f:
                data = json.load(f)
            self._tasks = data.get("tasks", [])
            # Compute TF per task
            self._tfs = []
            df: dict[str, int] = {}
            for task in self._tasks:
                tokens = _tokenize(task.get("instruction", "") + " " + task.get("tags", ""))
                tf = _tf(tokens)
                self._tfs.append(tf)
                for t in tf:
                    df[t] = df.get(t, 0) + 1
            # Compute IDF
            n = len(self._tasks) or 1
            self._idf = {t: math.log(n / (c + 1)) + 1 for t, c in df.items()}
            self._loaded = True
            logger.info("Task RAG loaded: %d tasks indexed.", len(self._tasks))
        except Exception as e:
            logger.error("Failed to load task index: %s", e)

    def _tfidf(self, tf: dict[str, float]) -> dict[str, float]:
        return {t: v * self._idf.get(t, 1.0) for t, v in tf.items()}

    def query(self, task_text: str, top_k: int = 2) -> str:
        """
        Returns scaffold-framed hint string for the top-k similar tasks.
        Returns empty string if RAG disabled or no good matches.
        """
        if not self._loaded or not self._tasks:
            return ""

        q_tokens = _tokenize(task_text)
        q_tfidf = self._tfidf(_tf(q_tokens))

        scores = [
            (i, _cosine(q_tfidf, self._tfidf(tf)))
            for i, tf in enumerate(self._tfs)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [(i, s) for i, s in scores[:top_k] if s > 0.15]  # min similarity threshold

        if not top:
            return ""

        hints = []
        for rank, (i, score) in enumerate(top, 1):
            task = self._tasks[i]
            hint = _format_hint(task, rank, score)
            if hint:
                hints.append(hint)

        return "\n".join(hints) if hints else ""


def _format_hint(task: dict, rank: int, score: float) -> str:
    """
    Format a task as a scaffold hint — process-oriented, not solution template.
    Critically: we do NOT dump the full oracle solution.
    We extract: domain, key tools used, known gotchas from solve.sh analysis.
    """
    name = task.get("name", "unknown")
    domain = task.get("domain", "")
    key_tools = task.get("key_tools", [])       # extracted at build time
    gotchas = task.get("gotchas", [])            # extracted at build time
    instruction_summary = task.get("summary", task.get("instruction", "")[:200])

    if not instruction_summary and not key_tools:
        return ""

    lines = [f"### Similar Task {rank} (similarity={score:.2f}): `{name}`"]
    if instruction_summary:
        lines.append(f"Pattern: {instruction_summary[:180]}")
    if domain:
        lines.append(f"Domain: {domain}")
    if key_tools:
        lines.append(f"Tools involved: {', '.join(key_tools[:8])}")
    if gotchas:
        lines.append("Known gotchas from similar tasks:")
        for g in gotchas[:3]:
            lines.append(f"  - {g}")
    lines.append("(Use this as a reasoning reference, not a copy-paste template.)")
    return "\n".join(lines)


# Singleton
_rag: TaskRAG | None = None


def get_rag() -> TaskRAG:
    global _rag
    if _rag is None:
        _rag = TaskRAG()
    return _rag


def query_rag(task_text: str, top_k: int = 2) -> str:
    """Module-level convenience function."""
    return get_rag().query(task_text, top_k=top_k)
