"""
Multi-label domain detection + specialist prompt fusion.

Design principles (anti-satiation):
  - Primary domain → full reasoning scaffold (diagnostic + pitfalls + anchors + 1 example)
  - Secondary domains → PITFALL WARNINGS ONLY (no examples, no templates)
  - Secondaries are capped at 3 to avoid context flooding
  - ICL examples show PROCESS not SOLUTION to prevent blind copying
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Keyword taxonomy with anchor overrides ──────────────────

# Anchor keywords: single hit → domain score boosted to 3 (above threshold=2)
_ANCHORS: dict[str, list[str]] = {
    "git":      ["git", "rebase", "squash", "cherry-pick", "reflog", "stash"],
    "docker":   ["docker", "dockerfile", "kubernetes", "container"],
    "python":   ["python", "python3", "pip", "virtualenv", "venv"],
    "database": ["sqlite", "sqlite3", "postgresql", "postgres", "psql", "mysql"],
    "network":  ["curl", "wget", "nginx", "apache", "http", "https"],
    "build":    ["makefile", "cmake", "cargo", "gcc", "clang"],
    "system":   ["systemctl", "crontab", "systemd", "chmod", "cron"],
    "text":     ["jq", "awk", "sed", "grep", "regex", "csv", "json"],
}

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "git":      ["git", "commit", "commits", "branch", "branches", "merge", "merged",
                 "rebase", "cherry-pick", "stash", "repository", "repo", "patch",
                 "diff", "blame", "tag", "remote", "origin", "HEAD", "reflog",
                 "squash", "amend", "push", "clone", "checkout", "bisect"],
    "docker":   ["docker", "container", "containers", "image", "images",
                 "dockerfile", "compose", "registry", "volume", "pod",
                 "kubernetes", "k8s", "entrypoint", "containerize", "build"],
    "python":   ["python", "python3", "pip", "pip3", "virtualenv", "venv",
                 "conda", "pytest", "pyproject", "requirements", "module",
                 "import", "script", "flask", "django", "fastapi", "package",
                 "wheel", "setuptools"],
    "database": ["sql", "sqlite", "sqlite3", "postgresql", "postgres", "mysql",
                 "mariadb", "database", "table", "query", "schema", "migration",
                 "index", "transaction", "psql", "pg_dump", "insert", "select"],
    "network":  ["curl", "wget", "http", "https", "api", "endpoint", "port",
                 "socket", "dns", "proxy", "ssl", "tls", "nginx", "apache",
                 "server", "client", "request", "response", "download",
                 "fetch", "url", "connect", "ssh", "netcat"],
    "build":    ["make", "makefile", "cmake", "gcc", "clang", "compile",
                 "compilation", "linker", "autoconf", "configure", "cargo",
                 "npm", "yarn", "webpack", "gradle", "maven", "bazel",
                 "binary", "executable"],
    "system":   ["systemctl", "service", "daemon", "cron", "crontab",
                 "schedule", "permission", "chmod", "chown", "sudo",
                 "process", "signal", "bashrc", "profile", "startup",
                 "boot", "init", "systemd", "ulimit"],
    "text":     ["csv", "json", "yaml", "xml", "parse", "extract",
                 "transform", "sed", "awk", "grep", "regex", "jq",
                 "logfile", "format", "convert", "encode", "decode",
                 "column", "delimiter"],
}

_THRESHOLD = 2


def _score(text: str) -> dict[str, int]:
    t = text.lower()
    scores: dict[str, int] = {}
    for domain, kws in _DOMAIN_KEYWORDS.items():
        count = 0
        for kw in kws:
            if " " in kw:
                count += t.count(kw)
            else:
                count += len(re.findall(r"\b" + re.escape(kw), t))
        if count > 0:
            scores[domain] = count

    # Anchor boost
    for domain, anchors in _ANCHORS.items():
        for anchor in anchors:
            if re.search(r"\b" + re.escape(anchor), t):
                scores[domain] = max(scores.get(domain, 0), 3)
                break
    return scores


@dataclass
class DomainResult:
    primary: str
    secondaries: list[str]   # up to 3, above threshold
    scores: dict[str, int]


def detect_domains(task_text: str) -> DomainResult:
    scores = _score(task_text)
    qualified = {d: s for d, s in scores.items() if s >= _THRESHOLD}
    if not qualified:
        return DomainResult(primary="generic", secondaries=[], scores=scores)
    ranked = sorted(qualified.items(), key=lambda x: x[1], reverse=True)
    primary = ranked[0][0]
    secondaries = [d for d, _ in ranked[1:4]]  # cap at 3
    return DomainResult(primary=primary, secondaries=secondaries, scores=scores)


# ── Prompt sections ─────────────────────────────────────────
# Each specialist has:
#   FULL  = diagnostics + pitfalls + anchors + 1 process example
#   SHORT = pitfalls only (injected for secondary domains)

_FULL: dict[str, str] = {}
_SHORT: dict[str, str] = {}

# ─── GIT ───────────────────────────────────────────────────
_FULL["git"] = """
## Git Specialist — Reasoning Scaffold

### How to Diagnose Git State (read before acting)
- `git log --oneline -15` → understand commit history depth and shape
- `git status` → staged vs unstaged vs untracked
- `git branch -a` → all local and remote branches
- `git stash list` → any stashed work that might matter
Read ALL output before forming a plan. The task solution depends on actual state.

### Common Failure Modes (understand WHY, not just what to avoid)
- `git rebase -i` opens $EDITOR interactively → process hangs forever.
  WHY: the exec endpoint has no TTY. Always use GIT_SEQUENCE_EDITOR non-interactively.
- Rebase mid-conflict leaves you in detached state. `git status` tells you exactly what to do next.
- Force-pushing when task says preserve remote history → read the task again carefully.
- Assuming HEAD~N without checking actual log depth → count commits first.

### Reasoning Anchors (ask yourself before every git command)
- Have I read `git log` output for this specific repo? (not assumed)
- Is this command safe without a TTY? (rebase -i, commit --amend need special handling)
- Am I assuming branch names? (checkout feature vs checkout main — verify with `git branch`)
- Will this destroy history the task cares about?

### Process Example (squash task)
Thought: Task says squash 3 commits. I need to know the actual commit SHAs first.
Command: git log --oneline -5
Observation: abc1234 fix typo / def5678 add tests / ghi9012 initial feature / ...
Thought: Three commits to squash. GIT_SEQUENCE_EDITOR needed for non-interactive rebase.
Command: GIT_SEQUENCE_EDITOR="sed -i '2,3s/^pick/squash/'" git rebase -i HEAD~3
Observation: [success] — check result
Command: git log --oneline -3
[verify expected state before proceeding]
"""
_SHORT["git"] = """
### Git Pitfalls (secondary domain awareness)
- Never run `git rebase -i` without setting GIT_SEQUENCE_EDITOR — it hangs (no TTY).
- Always read `git log --oneline` before any history-modifying command.
- Mid-rebase conflicts require `git rebase --continue` not a fresh rebase.
"""

# ─── DOCKER ────────────────────────────────────────────────
_FULL["docker"] = """
## Docker Specialist — Reasoning Scaffold

### How to Diagnose Docker State
- `docker images` → what's available locally
- `docker ps -a` → running and stopped containers
- `cat Dockerfile` → always read the full file before editing
- `docker build . 2>&1 | head -50` → first 50 lines of build output contain the error

### Common Failure Modes
- `COPY` path not found: build context doesn't include that path. Check `.dockerignore`.
- `RUN pip install` fails: base image missing build-essential or system deps.
- Port conflict: task container already occupies a port. Use `ss -tlnp` to check.
- `FROM latest` → unpinned, can break. But if task requires it, don't change it.
- Multi-stage build: artifact in stage 1 not copied to stage 2 → silent missing file.

### Reasoning Anchors
- Have I read the full Dockerfile before editing any line?
- Does the error occur at build or at runtime? (different fixes)
- Am I editing the right Dockerfile? (there may be multiple)
- Does the fix preserve the intent of the original author?

### Process Example (fix failing build)
Thought: Build is failing. Read error and Dockerfile first.
Command: docker build . 2>&1 | tail -30
Observation: ERROR: COPY requirements.txt not found
Command: ls -la
Observation: requirements.txt is named requirements-dev.txt
Thought: Filename mismatch. Fix the COPY line.
Command: sed -i 's/COPY requirements.txt/COPY requirements-dev.txt/' Dockerfile
Command: docker build . 2>&1 | tail -10
[verify build succeeds before done]
"""
_SHORT["docker"] = """
### Docker Pitfalls (secondary domain awareness)
- Always read the full Dockerfile before editing — errors are often elsewhere than they appear.
- Build errors vs runtime errors need different debugging approaches.
- `.dockerignore` silently excludes files from build context.
"""

# ─── PYTHON ────────────────────────────────────────────────
_FULL["python"] = """
## Python Specialist — Reasoning Scaffold

### How to Diagnose Python State
- `python3 --version` and `which python3` → confirm what's active
- `pip list 2>/dev/null | head -30` → installed packages
- `python3 -c "import pkg"` → fastest way to test an import
- Read tracebacks from BOTTOM UP — the root cause is the last line.

### Common Failure Modes
- ImportError: package not installed, wrong venv, or wrong python binary.
- SyntaxError in Python 3 from Python 2 syntax: `print "x"` → `print("x")`
- Relative imports failing outside package context: need `sys.path` or install.
- Script needs executable bit: `chmod +x script.py` AND correct shebang.
- `pip install` pollutes system python — task may expect venv isolation.

### Reasoning Anchors
- Which python binary does the task expect? (`python` vs `python3` vs venv)
- Is the error an import error or a logic error? (different root causes)
- Does the fix need to persist (install) or just work once (PYTHONPATH)?
- Have I read the script's imports and shebang line?

### Process Example (fix ImportError)
Thought: Script fails. Read the error message and the script first.
Command: python3 script.py 2>&1 | tail -10
Observation: ImportError: No module named 'requests'
Command: cat script.py | head -20
Thought: Script needs requests. Check if it should be in a venv or global.
Command: pip install requests 2>&1 | tail -5
Command: python3 script.py
[verify output matches expected]
"""
_SHORT["python"] = """
### Python Pitfalls (secondary domain awareness)
- Read tracebacks from BOTTOM UP — root cause is last.
- `which python3` before assuming any python binary.
- ImportError root causes: not installed, wrong binary, wrong venv.
"""

# ─── DATABASE ──────────────────────────────────────────────
_FULL["database"] = """
## Database Specialist — Reasoning Scaffold

### How to Diagnose Database State
- `file *.db *.sqlite* 2>/dev/null` → identify database files
- `sqlite3 db.db ".tables"` → schema overview
- `sqlite3 db.db ".schema tablename"` → exact column types
- For postgres: `pg_isready` before any connection attempt

### Common Failure Modes
- SQL single vs double quotes: strings need single quotes. `WHERE name = 'Alice'` not `"Alice"`.
- SQLite vs PostgreSQL syntax differences: `AUTOINCREMENT` vs `SERIAL`, etc.
- Trying to connect postgres before service is running: check `pg_isready` first.
- Missing semicolons in multi-statement SQL → silent partial execution.
- Overwriting existing data when task says append → read existing state first.

### Reasoning Anchors
- What database engine is this? (SQLite/Postgres/MySQL — different syntax)
- Does the table already exist? (schema inspection before CREATE)
- What is the exact expected schema? (read task requirements carefully)
- Will my INSERT conflict with existing rows?

### Process Example (create and populate table)
Thought: Task needs a users table. Check if db file exists and what's in it.
Command: ls -la *.db *.sqlite3 2>/dev/null
Command: sqlite3 app.db ".tables"
Observation: (empty — no tables yet)
Command: sqlite3 app.db "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
Command: sqlite3 app.db "INSERT INTO users (name) VALUES ('Alice'), ('Bob');"
Command: sqlite3 app.db "SELECT * FROM users;"
[verify output matches task requirements]
"""
_SHORT["database"] = """
### Database Pitfalls (secondary domain awareness)
- SQL strings need single quotes, not double quotes.
- Check `pg_isready` before postgres connections.
- Inspect existing schema before CREATE or INSERT.
"""

# ─── NETWORK ───────────────────────────────────────────────
_FULL["network"] = """
## Network Specialist — Reasoning Scaffold

### How to Diagnose Network State
- `ss -tlnp` → what ports are listening and which process owns them
- `curl -sv url 2>&1 | head -40` → full HTTP negotiation including redirects
- `cat /etc/resolv.conf` → DNS config if resolution is failing
- `env | grep -i proxy` → proxy settings that may interfere

### Common Failure Modes
- Missing `-L` flag: curl doesn't follow redirects by default. `curl -L url`.
- Auth headers: API endpoints need `Authorization: Bearer token` — read task for credentials.
- JSON body without Content-Type header → server rejects silently with 415.
- Service not started: curl to localhost fails because process isn't running yet.
- Certificate errors with `-k` flag: only acceptable if task explicitly says so.

### Reasoning Anchors
- Is the target service actually running? (`ss -tlnp` or `curl localhost:port/health`)
- Does this request need authentication?
- Should I follow redirects? (most downloads: yes)
- What does a successful response look like? (status code AND body)

### Process Example (download and verify)
Thought: Need to download file. Check connectivity and get exact URL from task.
Command: curl -I "https://example.com/file.tar.gz" 2>&1 | head -10
Observation: HTTP/2 200 — good, file exists
Command: curl -L -o artifact.tar.gz "https://example.com/file.tar.gz"
Command: echo "expectedhash  artifact.tar.gz" | sha256sum -c
[verify checksum passes]
"""
_SHORT["network"] = """
### Network Pitfalls (secondary domain awareness)
- Always use `curl -L` for downloads (follows redirects).
- Check if the target service is running before connecting.
- JSON POST requests need Content-Type: application/json header.
"""

# ─── BUILD ─────────────────────────────────────────────────
_FULL["build"] = """
## Build System Specialist — Reasoning Scaffold

### How to Diagnose Build State
- `cat Makefile | head -40` → understand targets and dependencies
- `make -n target` → dry run, see what would execute without running it
- `make target 2>&1 | head -30` → first error is usually the root cause
- `ldd binary` → check shared library dependencies after build

### Common Failure Modes
- Missing header: `-I/path/to/include` needed. Check what package provides it.
- Missing library: `-L/path -lname` needed, or `apt-get install libname-dev`.
- `make` assumes tab indentation — spaces cause "missing separator" error.
- Parallel build (`-j4`) hides ordering issues that sequential build would catch.
- CMake out-of-source build: `cmake -B build -S .` then `cmake --build build`.

### Reasoning Anchors
- Have I read the Makefile/CMakeLists.txt before running anything?
- Is this a missing header, missing library, or missing tool?
- Does the build system expect to be run from a specific directory?
- Are environment variables like CC, CXX, PKG_CONFIG_PATH relevant?

### Process Example (fix compile error)
Thought: Build fails. Read error and Makefile first.
Command: make 2>&1 | grep -E "error:|Error" | head -10
Observation: fatal error: openssl/ssl.h: No such file or directory
Thought: Missing OpenSSL headers. Check if dev package is installed.
Command: apt-get install -y libssl-dev 2>&1 | tail -5
Command: make 2>&1 | tail -10
[verify build succeeds]
"""
_SHORT["build"] = """
### Build Pitfalls (secondary domain awareness)
- Read Makefile before running make — understand targets first.
- Missing headers need dev packages (`libname-dev`), not just runtime packages.
- CMake needs out-of-source build directory.
"""

# ─── SYSTEM ────────────────────────────────────────────────
_FULL["system"] = """
## System Administration Specialist — Reasoning Scaffold

### How to Diagnose System State
- `systemctl status service` → is it running, what errors
- `crontab -l` → existing cron jobs (don't overwrite silently)
- `ls -la /etc/cron.d/ /etc/cron.daily/` → system-level cron
- `id && whoami` → current user and permissions context

### Common Failure Modes
- systemctl in containers: systemd often not running. Check `ps aux | grep systemd`.
  Alternative: start service directly or use `/etc/init.d/service start`.
- Cron requires executable script AND correct PATH in crontab environment.
- chmod 755 needed for scripts to be executable by cron.
- `/etc/environment` changes don't take effect in current shell — need new session.
- File permission denied: check both file permissions AND directory traversal (+x on dirs).

### Reasoning Anchors
- Is systemd actually running in this container? (check before systemctl)
- What user will run this (cron, service)? Different from current user.
- Does the existing crontab need preserving? (always `crontab -l` first)
- Are the required paths absolute? (cron has minimal PATH)

### Process Example (add cron job)
Thought: Need to add cron job. Check existing crontab first.
Command: crontab -l 2>/dev/null || echo "(empty)"
Observation: (empty)
Command: chmod +x /usr/local/bin/job.sh
Command: (crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/job.sh >> /var/log/job.log 2>&1") | crontab -
Command: crontab -l
[verify job appears correctly]
"""
_SHORT["system"] = """
### System Pitfalls (secondary domain awareness)
- systemd may not be running in containers — verify before systemctl.
- Always `crontab -l` before adding jobs — don't overwrite existing ones.
- Cron scripts need absolute paths and executable permissions.
"""

# ─── TEXT ──────────────────────────────────────────────────
_FULL["text"] = """
## Text Processing Specialist — Reasoning Scaffold

### How to Diagnose Text/Data State
- `file unknown_file` → identify actual format before parsing
- `head -5 file` → see structure before writing any transform
- `wc -l file` → understand scale before choosing approach
- `cat file | python3 -m json.tool` → validate JSON structure

### Common Failure Modes
- sed regex: `.` matches any char, escape with `\\.` for literal dot.
- `sed -i` on Linux works; on macOS requires `sed -i ''` — we're in containers, Linux only.
- CSV with quoted commas: `cut -d,` fails. Use Python csv module or awk with FPAT.
- YAML parsing: tabs are invalid in YAML — use spaces only.
- `grep -P` for Perl regex; plain `grep` doesn't support `\\d`, `\\w`, etc.

### Reasoning Anchors
- Have I looked at actual file contents before writing a transform?
- Is the delimiter consistent throughout the file?
- Does the regex handle edge cases (empty fields, special chars)?
- Will this transform work on the full file or just the head sample?

### Process Example (extract emails from log)
Thought: Need emails from log. Check file structure first.
Command: head -3 logfile.log
Observation: 2024-01-01 user@example.com logged in
Thought: Space-delimited. Email is field 2. Use grep with regex.
Command: grep -oP '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+[.][a-zA-Z]{2,}' logfile.log | sort -u
[verify count and format of output]
"""
_SHORT["text"] = """
### Text Pitfalls (secondary domain awareness)
- Always inspect actual file contents before writing any transform.
- `grep -P` for Perl regex; plain grep doesn't support `\\d`, `\\w`.
- CSV with quoted commas breaks simple `cut -d,` — use Python or awk FPAT.
"""

# ─── GENERIC ───────────────────────────────────────────────
_FULL["generic"] = """
## General Terminal Task — Reasoning Scaffold

### Universal Orientation Protocol (always start here)
1. `pwd && ls -la` — where am I, what's here
2. `find . -maxdepth 2 -type f | sort` — full file inventory
3. Read any README, task instructions, or config files present
4. Identify the exact success condition from the task description
5. Plan sub-goals before issuing any modifying command

### Diagnostic Mindset
- What is the CURRENT state? (explore before acting)
- What is the DESIRED state? (re-read task if unclear)
- What is the SMALLEST change that gets from current to desired?
- How will I VERIFY the change worked?

### Common Cross-Domain Failures
- Assuming file paths without checking: always `ls` first
- Running commands as root when task requires specific user (check `id`)
- Modifying files that shouldn't be touched (read task constraints carefully)
- Declaring done without verifying the actual end state

### Reasoning Anchors
- Have I re-read the task description recently?
- Is my current approach actually working toward the stated goal?
- What would the verifier check? (think backwards from success)
- Are there simpler approaches I haven't considered?
"""
_SHORT["generic"] = ""  # Not used as secondary


def build_system_prompt(
    domain_result: DomainResult,
    rag_hints: str = "",
    max_turns: int = 30,
) -> str:
    """
    Assemble the full system prompt.
    Primary domain: full scaffold.
    Secondary domains: pitfall sections only (anti-satiation).
    RAG hints: scaffold-framed oracle patterns.
    """
    base = f"""You are an expert terminal agent solving hard, realistic command-line tasks.
You operate inside a Linux container with full shell access via an exec endpoint.

## Core ReAct Protocol
You will be given a PLAN of sub-goals to accomplish in order.
For each sub-goal, you reason then act ONE COMMAND AT A TIME.

## Response Format
ALWAYS respond with exactly one of these structures:

For acting:
<thought>
Step-by-step reasoning grounded in what you actually observed.
Reference specific files, paths, and values from real output — never assume.
</thought>
<command>
single_bash_command_here
</command>

For completing the task (only after self-verification passes):
<thought>
State what you verified and how you confirmed success.
</thought>
<done>
Brief factual summary of what was accomplished.
</done>

## Non-Negotiable Rules
- ONE command per response. No command chaining with unrelated operations.
- Ground every reasoning step in OBSERVED output, not assumed state.
- Never hallucinate file contents — read with cat/head/grep first.
- Non-zero exit codes MUST be addressed before proceeding.
- You have {max_turns} total turns. **HARD BUDGET — stick to it or fail.**
  - Turn 0: RECON (already done — free)
  - Turn 1: Plan injected (free)
  - Turns 2-{max_turns - 8}: EXECUTE subgoals (max 3-4 turns per subgoal)
  - Turns {max_turns - 5}-{max_turns}: VERIFY and declare done
- If you are on turn >={max_turns - 3} and not done: STOP exploring, WRITE a best-effort solution NOW and declare done.
- A partial solution that attempts the task scores higher than running out of turns with nothing written.
- Never spend more than 3 consecutive turns on the same sub-goal — if stuck, move on.
- The verifier checks the FINAL container state. Think backwards from that.

## CRITICAL: 30-SECOND HARD TIMEOUT PER COMMAND
Every command has a HARD 30-second timeout. If it exceeds 30 seconds the ENTIRE TASK FAILS immediately.

Commands that WILL timeout (avoid these):
- apt-get update alone takes 15-20s, leaves no room for install
- apt-get update followed by apt-get install always times out
- pip install of large packages (torch, tensorflow, easyocr) takes 60-300s
- make/cargo build/npm install on large projects
- find / or find with no depth limit

Safe package installation strategy:
1. Check first: python3 -c "import X" 2>/dev/null && echo OK || echo MISSING
2. Use pip install --break-system-packages PACKAGE 2>&1 | tail -3 (safe in Docker)
3. For apt: skip apt-get update, just run apt-get install -y PACKAGE 2>&1 | tail -5
4. Install ONE small package at a time
5. If a package cannot be installed in time, solve the task without it
"""

    sections = [base, _FULL.get(domain_result.primary, _FULL["generic"])]

    for sec_domain in domain_result.secondaries:
        pitfall = _SHORT.get(sec_domain, "")
        if pitfall:
            sections.append(pitfall)

    if rag_hints:
        sections.append(f"\n## Retrieved Patterns From Similar Tasks\n{rag_hints}")

    return "\n".join(sections)