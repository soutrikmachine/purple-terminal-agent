"""
Unit tests for Purple Terminal Agent components.
All tests are pure-Python with no network or Docker dependencies.
"""

from __future__ import annotations

import json
import sys
import pytest

sys.path.insert(0, "src")


# ── Specialist / Domain Detection ──────────────────────────

class TestDomainDetection:
    def setup_method(self):
        from specialist import detect_domains
        self.detect = detect_domains

    def test_single_anchor_git(self):
        r = self.detect("Squash the last 3 git commits into one.")
        assert r.primary == "git"

    def test_single_anchor_docker(self):
        r = self.detect("Fix the Dockerfile so the docker build succeeds.")
        assert r.primary == "docker"

    def test_single_anchor_python(self):
        r = self.detect("Fix the python3 script so pip install works correctly.")
        assert r.primary == "python"

    def test_single_anchor_database(self):
        r = self.detect("Create a sqlite3 database with a users table.")
        assert r.primary == "database"

    def test_single_anchor_network(self):
        r = self.detect("Download the file using curl and verify its checksum.")
        assert r.primary == "network"

    def test_generic_fallback(self):
        r = self.detect("Set up the environment.")
        assert r.primary == "generic"

    def test_multi_domain_git_python(self):
        r = self.detect(
            "Write a python3 script to process files, commit it with git, "
            "and rebase the feature branch onto main."
        )
        assert r.primary in ("git", "python")
        assert len(r.secondaries) >= 1

    def test_secondaries_capped_at_3(self):
        r = self.detect(
            "Use git, docker, python3, sqlite3, curl, and make to build the system."
        )
        assert len(r.secondaries) <= 3

    def test_rebase_anchor(self):
        r = self.detect("Rebase the feature branch onto main and resolve conflicts.")
        assert r.primary == "git"

    def test_nginx_anchor(self):
        r = self.detect("Configure nginx to proxy requests to the backend service.")
        assert r.primary == "network"


class TestPromptBuilding:
    def test_primary_full_contains_diagnostic(self):
        from specialist import detect_domains, build_system_prompt
        r = detect_domains("Fix the git rebase conflict.")
        prompt = build_system_prompt(r, max_turns=25)
        assert "Diagnose" in prompt or "diagnose" in prompt.lower() or "git log" in prompt

    def test_secondary_pitfalls_only(self):
        from specialist import DomainResult, build_system_prompt
        r = DomainResult(primary="python", secondaries=["git"], scores={})
        prompt = build_system_prompt(r, max_turns=30)
        # Git should appear as pitfalls, not full example
        assert "Git Pitfalls" in prompt
        # But not the full git specialist header
        assert "Git Specialist — Reasoning Scaffold" not in prompt

    def test_rag_hints_included(self):
        from specialist import detect_domains, build_system_prompt
        r = detect_domains("some task")
        prompt = build_system_prompt(r, rag_hints="## Retrieved Patterns\nHint 1", max_turns=30)
        assert "Retrieved Patterns" in prompt

    def test_max_turns_in_prompt(self):
        from specialist import detect_domains, build_system_prompt
        r = detect_domains("some task")
        prompt = build_system_prompt(r, max_turns=25)
        assert "25" in prompt


# ── Exec URL Extraction ────────────────────────────────────

class TestExecUrlExtraction:
    def setup_method(self):
        from agent import extract_exec_url
        self.extract = extract_exec_url

    def test_standard_format(self):
        assert self.extract("exec_url: http://host:9010/exec/abc123") == "http://host:9010/exec/abc123"

    def test_post_format(self):
        assert self.extract("use POST http://10.0.0.1:9010/exec/tok run it") == "http://10.0.0.1:9010/exec/tok"

    def test_inline_url(self):
        assert self.extract("shell access at http://green:9010/exec/xyz999") == "http://green:9010/exec/xyz999"

    def test_trailing_punctuation_stripped(self):
        url = self.extract("exec_url: http://host:9010/exec/abc.")
        assert url == "http://host:9010/exec/abc"

    def test_none_when_missing(self):
        assert self.extract("Fix the bug in the repository. No URL.") is None

    def test_exec_endpoint_format(self):
        url = self.extract("exec_endpoint: http://amber:9010/exec/session42")
        assert url is not None
        assert "session42" in url


# ── Critic Fast Checks ─────────────────────────────────────

class TestCriticFastCheck:
    def setup_method(self):
        from critic import _fast_check
        self.check = _fast_check

    def test_bare_rebase_flagged(self):
        needs, issue, fix = self.check("git rebase -i HEAD~3")
        assert needs is True
        assert "GIT_SEQUENCE_EDITOR" in issue or "interactive" in issue.lower()

    def test_rebase_with_editor_approved(self):
        needs, _, _ = self.check("GIT_SEQUENCE_EDITOR='sed -i s/pick/squash/' git rebase -i HEAD~3")
        assert needs is False

    def test_vim_flagged(self):
        needs, issue, _ = self.check("vim config.yaml")
        assert needs is True

    def test_git_commit_without_m_flagged(self):
        needs, _, fix = self.check("git commit")
        assert needs is True

    def test_git_commit_with_m_approved(self):
        needs, _, _ = self.check("git commit -m 'fix: update logic'")
        assert needs is False

    def test_apt_without_y_flagged(self):
        needs, _, fix = self.check("apt-get install python3-dev")
        assert needs is True

    def test_apt_with_y_approved(self):
        # Now requires BOTH the -y flag (interactive safety) 
        # AND output bounding (504 Gateway API safety)
        needs, _, _ = self.check("apt-get install -y python3-dev > /tmp/out 2>&1; tail -n 10 /tmp/out")
        assert needs is False

    def test_ls_is_safe(self):
        needs, _, _ = self.check("ls -la /workspace")
        assert needs is False

    def test_git_log_is_safe(self):
        needs, _, _ = self.check("git log --oneline -10")
        assert needs is False

    def test_git_merge_without_no_edit_flagged(self):
        needs, _, fix = self.check("git merge feature-branch")
        assert needs is True


# ── Clearly Safe Detection ─────────────────────────────────

class TestClearlySafe:
    def setup_method(self):
        from critic import _is_clearly_safe
        self.safe = _is_clearly_safe

    def test_ls_safe(self):       assert self.safe("ls -la")
    def test_cat_safe(self):      assert self.safe("cat config.yaml")
    def test_git_log_safe(self):  assert self.safe("git log --oneline -5")
    def test_git_status_safe(self): assert self.safe("git status")
    def test_rm_not_safe(self):   assert not self.safe("rm -rf node_modules")
    def test_pip_install_not_safe(self): assert not self.safe("pip install requests")


# ── ExecResult ─────────────────────────────────────────────

class TestExecResult:
    def test_stdout_only(self):
        from executor import ExecResult
        r = ExecResult("hello world", "", 0)
        assert r.combined == "hello world"
        assert r.success

    def test_stderr_included(self):
        from executor import ExecResult
        r = ExecResult("out", "err msg", 1)
        assert "out" in r.combined
        assert "err msg" in r.combined
        assert not r.success

    def test_empty_shows_exit_code(self):
        from executor import ExecResult
        r = ExecResult("", "", 0)
        assert "0" in r.combined

    def test_truncation_in_combined(self):
        from executor import ExecResult, MAX_OUTPUT_CHARS
        big = "X" * (MAX_OUTPUT_CHARS + 1000)
        r = ExecResult(big, "", 0)
        assert "omitted" in r.combined


# ── RAG TF-IDF ─────────────────────────────────────────────

class TestTFIDF:
    def test_tokenize(self):
        from rag import _tokenize
        tokens = _tokenize("Fix the git rebase conflict in repository")
        assert "git" in tokens
        assert "rebase" in tokens
        assert "the" not in tokens  # stopword

    def test_cosine_identical(self):
        from rag import _cosine
        v = {"git": 0.5, "rebase": 0.3}
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_cosine_disjoint(self):
        from rag import _cosine
        a = {"git": 0.5}
        b = {"docker": 0.3}
        assert _cosine(a, b) == 0.0

    def test_rag_empty_index_returns_empty(self):
        from rag import TaskRAG
        rag = TaskRAG.__new__(TaskRAG)
        rag._tasks = []
        rag._tfs = []
        rag._idf = {}
        rag._loaded = False
        result = rag.query("fix git commits", top_k=2)
        assert result == ""

    def test_rag_with_mock_tasks(self):
        from rag import TaskRAG, _tf
        rag = TaskRAG.__new__(TaskRAG)
        rag._tasks = [
            {"name": "fix-git", "instruction": "squash git commits rebase", "tags": "git",
             "summary": "Squash git commits", "key_tools": ["git"], "gotchas": [], "domain": "git"},
            {"name": "fix-docker", "instruction": "fix docker build dockerfile", "tags": "docker",
             "summary": "Fix docker build", "key_tools": ["docker"], "gotchas": [], "domain": "docker"},
        ]
        import math
        rag._tfs = [
            _tf(["squash", "git", "commits", "rebase"]),
            _tf(["fix", "docker", "build", "dockerfile"]),
        ]
        df = {}
        for tf in rag._tfs:
            for t in tf:
                df[t] = df.get(t, 0) + 1
        n = len(rag._tasks)
        rag._idf = {t: math.log(n / (c + 1)) + 1 for t, c in df.items()}
        rag._loaded = True

        result = rag.query("squash the last 3 git commits", top_k=1)
        assert "fix-git" in result or "git" in result.lower()


# ── Memory ─────────────────────────────────────────────────

class TestTaskMemory:
    def test_unverified_not_stored(self):
        from memory import TaskMemory
        m = TaskMemory()
        m.store("git", "task", ["git log"], "saw commits", verified=False)
        assert m.retrieve("git") == []

    def test_verified_stored(self):
        from memory import TaskMemory
        m = TaskMemory()
        m.store("git", "task text", ["git log --oneline -5", "git rebase ..."], "squashed", verified=True)
        seqs = m.retrieve("git")
        assert len(seqs) == 1
        assert "git log" in seqs[0].commands[0]

    def test_cap_at_max(self):
        from memory import TaskMemory, MAX_PER_DOMAIN
        m = TaskMemory()
        for i in range(MAX_PER_DOMAIN + 5):
            m.store("python", f"task {i}", [f"cmd {i}"], "ok", verified=True)
        # Check internal bucket (storage cap), not retrieve() which has its own MAX_INJECT default
        internal = len(m._store.get("python", []))
        assert internal == MAX_PER_DOMAIN

    def test_format_non_empty(self):
        from memory import TaskMemory
        m = TaskMemory()
        m.store("docker", "build task", ["docker build ."], "built ok", verified=True)
        out = m.format_for_injection("docker")
        assert "docker build" in out
        assert "REASONING REFERENCE" in out or "reasoning reference" in out.lower()

    def test_format_empty_domain(self):
        from memory import TaskMemory
        m = TaskMemory()
        assert m.format_for_injection("system") == ""


# ── Planner Output Formatting ──────────────────────────────

class TestPlannerFormatting:
    def test_format_plan(self):
        from planner import format_plan_for_executor
        plan = {
            "understanding": "Squash commits",
            "success_condition": "Single commit on HEAD",
            "risks": ["may lose history"],
            "subgoals": [
                {"id": 1, "goal": "Check git log", "domain": "git", "verification": "see N commits", "estimated_turns": 2},
                {"id": 2, "goal": "Squash with rebase", "domain": "git", "verification": "one commit", "estimated_turns": 3},
            ],
        }
        result = format_plan_for_executor(plan)
        assert "Squash commits" in result
        assert "[1]" in result
        assert "[2]" in result
        assert "may lose history" in result

    def test_fallback_plan_valid(self):
        from planner import _fallback_plan
        plan = _fallback_plan("Fix the git repository")
        assert "subgoals" in plan
        assert len(plan["subgoals"]) >= 2
        for sg in plan["subgoals"]:
            assert "goal" in sg
            assert "id" in sg


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])