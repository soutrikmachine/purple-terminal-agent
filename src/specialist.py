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
    "security": ["xss", "injection", "exploit", "cryptanalysis", "cipher",
                 "hash", "crack", "payload", "bypass", "vulnerability"],
    "ml":       ["pytorch", "torch", "tensorflow", "huggingface", "transformers",
                 "bert", "gpt", "cuda", "fine-tune", "checkpoint", "embedding"],
    "scientific": ["bayesian", "mcmc", "sampling", "scipy", "numpy", "pandas",
                   "statistics", "regression", "distribution", "parquet",
                   "pgmpy", "stan", "rscript", "raman", "spectroscopy",
                   "protein", "fasta", "dna"],
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
    "security": ["xss", "injection", "sql injection", "exploit", "payload",
                 "bypass", "vulnerability", "attack", "cipher", "encrypt",
                 "decrypt", "hash", "crack", "cryptanalysis", "chosen plaintext",
                 "differential", "linear", "brute force", "sanitize", "sanitise",
                 "secret", "credential", "password", "authentication", "jwt",
                 "token", "session", "csrf", "reverse engineer", "disassemble",
                 "obfuscate", "malware", "shellcode", "overflow", "memory"],
    "ml":       ["pytorch", "torch", "tensorflow", "keras", "huggingface",
                 "transformers", "bert", "gpt", "llm", "model", "training",
                 "inference", "neural", "deep learning", "cuda", "gpu",
                 "batch", "epoch", "loss", "gradient", "weights", "checkpoint",
                 "fine-tune", "finetune", "embedding", "tokenizer", "dataset",
                 "dataloader", "optimizer", "scheduler", "attention", "forward",
                 "backward", "autograd", "tensor", "pipeline", "parallelism",
                 "caffe", "fasttext", "sam", "segment", "diffusion"],
    "scientific": ["r language", "rscript", "bayesian", "mcmc", "sampling",
                   "scipy", "numpy", "pandas", "statsmodels", "pgmpy",
                   "statistics", "statistical", "regression", "correlation",
                   "distribution", "probability", "prior", "posterior",
                   "parquet", "arrow", "dask", "feather", "hdf5",
                   "spectroscopy", "fitting", "optimization", "simulation",
                   "ode", "pde", "numerical", "scientific", "notebook",
                   "matplotlib", "seaborn", "plotly", "visualization"],
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

# ─── SECURITY ──────────────────────────────────────────────
_FULL["security"] = """
## Security Specialist — Reasoning Scaffold

### How to Diagnose the Security Environment
- `cat /app/filter.py` or the relevant script FIRST — understand what it does before trying to bypass
- `file /app/*.py /app/*.c /app/*.js 2>/dev/null` → identify what code is present
- `strings binary | grep -iE "key|secret|password|token"` → embedded secrets in binaries
- `git log --all --oneline` + `git diff SHA1 SHA2` → find removed secrets in history
- `nm binary` or `objdump -d binary | head -100` → understand binary structure

### Task Archetypes and Approaches

**XSS / HTML filter bypass** (`break-filter-js-from-html`):
- Read the filter CAREFULLY — understand exactly what it blocks and what it misses
- Think about encoding bypasses: HTML entities, URL encoding, Unicode, case variations
- Think about event handler attributes: `onerror`, `onload`, `onfocus`, `onmouseover`
- Think about SVG, MathML, template tags, CSS expressions — non-script JS execution vectors
- Test: `python3 /app/filter.py /app/out.html` then check what survives
- NEVER assume the bypass — test it against the actual filter

**Cryptanalysis** (`feal-linear-cryptanalysis`, `feal-differential-cryptanalysis`):
- Read the cipher implementation COMPLETELY before writing attack code
- Write the attack as a Python script — express mathematical operations as code, NOT prose
- Differential: collect (plaintext, ciphertext) pairs, find key bits from XOR differentials
- Linear: collect many pairs, build linear approximations, recover key bits by majority vote
- Test with known values first: verify cipher encryption/decryption before the attack
- Key workflow: `cat /app/feal.c` → understand rounds → write attack.py → test on small examples

**Hash cracking** (`crack-7z-hash`):
- `hashcat --help | grep 7z` or `john --list=formats | grep 7z` → find the right format
- Extract hash: `7z2john archive.7z > hash.txt` (install with `pip install 7z2john`)
- `hashcat -m 11600 hash.txt wordlist.txt` or `john hash.txt --wordlist=/usr/share/wordlists/rockyou.txt`
- Common wordlists: `/usr/share/wordlists/`, `/usr/share/dict/words`

**Secret sanitization** (`sanitize-git-repo`, `git-leak-recovery`):
- `git log --all --full-history --diff-filter=D -- "*.env" "*.key" "*secret*"` → find deleted files
- `git grep -E "secret|password|api_key" $(git log --all --pretty=%H)` → search all history
- BFG Repo Cleaner: faster than `git filter-branch` for removing secrets
- Always verify with `git log --all --oneline` after cleanup

**Code vulnerability fixing** (`fix-code-vulnerability`, `vulnerable-secret`):
- `grep -rn -E "eval|exec|system|popen|subprocess" /app/` → find dangerous calls
- `grep -rn -E "TODO|FIXME|password|secret|api_key" /app/` → find hardcoded secrets
- Read the existing code COMPLETELY before patching — understand context
- Test the fix: re-run the test harness if one exists

### Reasoning Anchors
- Have I read the target code/binary COMPLETELY before attempting the attack/fix?
- For filter bypass: have I tested my payload against the ACTUAL filter, not my mental model?
- For cryptanalysis: am I expressing the attack as EXECUTABLE CODE, not prose analysis?
- For hash cracking: do I know the EXACT hash format before choosing a tool?
- Would the task verifier accept my output format? (check exact expected file paths/formats)

### Process Example (FEAL cryptanalysis)
Thought: Task asks for chosen plaintext attack. Read cipher implementation first.
Command: cat /app/feal.c
Thought: FEAL-4 with 4 rounds. Write attack script targeting key[5] via differential pairs.
Command: python3 << 'EOF'
# Quick sanity check: encrypt known plaintext
import subprocess
result = subprocess.run(['python3', '-c', 'import sys; sys.path.insert(0,"/app"); from feal import encrypt; print(encrypt(b"\\x00"*8, [0]*6))'], capture_output=True, text=True)
print(result.stdout, result.stderr)
EOF
Thought: Cipher verified. Now write full differential attack.
Command: cat > /app/attack.py << 'SCRIPT'
# Full differential attack code here
SCRIPT
Command: python3 /app/attack.py
[Verify key[5] value matches expected]
"""
_SHORT["security"] = """
### Security Pitfalls (secondary domain awareness)
- For filter bypasses: test against the ACTUAL filter, not assumptions about what it blocks.
- For cryptanalysis: write the attack as Python code — mathematical prose without <command> does nothing.
- For secret removal: always `git log --all --oneline` BEFORE and AFTER to verify cleanup.
- For code vulnerabilities: read the FULL code before patching — don't patch blindly.
"""

# ─── ML ────────────────────────────────────────────────────
_FULL["ml"] = """
## Machine Learning Specialist — Reasoning Scaffold

### How to Diagnose the ML Environment
- `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → PyTorch + GPU
- `python3 -c "import transformers; print(transformers.__version__)"` → HuggingFace
- `nvidia-smi 2>/dev/null || echo 'No GPU'` → GPU availability
- `pip list 2>/dev/null | grep -E "torch|tensorflow|transformers|numpy"` → installed ML stack
- `ls -lh /app/*.pt /app/*.pth /app/*.ckpt /app/*.bin 2>/dev/null` → existing model files
- `ls -lh /app/data/ /app/dataset/ ~/.cache/huggingface/ 2>/dev/null` → data and model cache

### Task Archetypes and Approaches

**Model inference** (`hf-model-inference`, `pytorch-model-cli`):
- ALWAYS check if model files exist locally before downloading anything
- `python3 -c "from transformers import AutoModel; m = AutoModel.from_pretrained('/app/model')"` → test load
- For HuggingFace: cache is at `~/.cache/huggingface/` — check before downloading again
- Batch size 1 for inference unless task specifies; use `torch.no_grad()` context

**Model recovery/repair** (`pytorch-model-recovery`):
- `python3 -c "import torch; d = torch.load('/app/model.pt', map_location='cpu'); print(type(d), list(d.keys()) if isinstance(d,dict) else 'tensor')"` → inspect checkpoint
- Missing keys vs unexpected keys in `load_state_dict` → architecture mismatch, fix manually
- `strict=False` in `load_state_dict` to load partial weights: `model.load_state_dict(state, strict=False)`

**Parallelism** (`torch-pipeline-parallelism`, `torch-tensor-parallelism`):
- Without GPU: pipeline parallelism uses CPU devices (`device_map="cpu"` or manual `.to("cpu:0")`)
- Tensor parallelism: split weight matrices across devices — verify with dummy forward pass
- Always test with a small input before claiming success

**Training** (`caffe-cifar-10`, `train-fasttext`):
- Read existing config files COMPLETELY before modifying anything
- `caffe train --solver=/path/to/solver.prototxt 2>&1 | tee /app/training_output.txt`
- fastText: `pip install fasttext-wheel` or `pip install fasttext` (not `fasttext-python`)
- For large training: redirect output to file and tail it — don't run blocking in foreground

**Dataset tasks** (`count-dataset-tokens`, `reshard-c4-data`, `mteb-leaderboard`):
- HuggingFace datasets: `from datasets import load_dataset; ds = load_dataset("name", split="train")`
- Tokenizer: `from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("model")`
- For large datasets: use `.map()` with `batched=True` for efficiency, not element-wise loops
- MTEB: `pip install mteb; python3 -c "from mteb import MTEB; print(MTEB().tasks)"` → list tasks

**Image/segmentation** (`sam-cell-seg`, `code-from-image`):
- SAM: `pip install segment-anything` then download checkpoint from Meta
- `from PIL import Image; img = Image.open('/app/image.png'); print(img.size, img.mode)` → inspect first
- For OCR: tesseract (`apt-get install -y tesseract-ocr`) or easyocr (`pip install easyocr`)

### Reasoning Anchors
- Is the required model/checkpoint already cached locally? (check before downloading)
- Does the task need GPU? (most containers are CPU-only — use `map_location='cpu'`)
- Am I reading existing config files before modifying? (don't write from scratch)
- What exact output format does the verifier expect? (file name, format, precision)
- Is the pip install output-bounded? (large models like torch can take minutes)

### Process Example (HuggingFace inference)
Thought: Need to run inference. Check what's installed and what model files exist.
Command: python3 -c "import torch,transformers; print(torch.__version__, transformers.__version__)"
Command: ls -lh /app/model/ ~/.cache/huggingface/ 2>/dev/null
Thought: Model exists locally. Load and run inference.
Command: python3 << 'EOF'
from transformers import pipeline
pipe = pipeline("text-classification", model="/app/model", device=-1)
result = pipe("test input text")
print(result)
EOF
[Verify output format matches task requirements]
"""
_SHORT["ml"] = """
### ML Pitfalls (secondary domain awareness)
- Always check if model files exist locally before downloading (check ~/.cache/huggingface/).
- Most containers are CPU-only: use `map_location='cpu'` and `device=-1`.
- Large pip installs (torch, tensorflow) need output bounding: `> /tmp/pip.log 2>&1; tail -5 /tmp/pip.log`
- For training tasks: redirect output to the required file, don't assume console output is captured.
"""

# ─── SCIENTIFIC ─────────────────────────────────────────────
_FULL["scientific"] = """
## Scientific Computing Specialist — Reasoning Scaffold

### How to Diagnose the Scientific Environment
- `python3 -c "import numpy,scipy,pandas; print(numpy.__version__, scipy.__version__, pandas.__version__)"` → core stack
- `Rscript --version 2>/dev/null || echo 'R not available'` → R availability
- `python3 -c "import pgmpy; print(pgmpy.__version__)"` 2>/dev/null → Bayesian network library
- `head -3 /app/*.csv /app/*.parquet /app/*.fasta 2>/dev/null` → inspect data format

### Task Archetypes and Approaches

**Bayesian networks** (`bn-fit-modify`):
- Library: `pgmpy` (Bayesian networks) — `pip install --break-system-packages pgmpy`
- Workflow: load CSV → run structure learning → fit parameters → perform intervention → sample
- `from pgmpy.estimators import PC, HillClimbSearch; from pgmpy.models import BayesianNetwork`
- Causal intervention: `model.do()` or set CPD with Dirac delta at intervention value
- Save edges: `pd.DataFrame(list(model.edges()), columns=['to','from']).to_csv('/app/learned_dag.csv', index=False)`

**Statistical sampling and distributions** (`adaptive-rejection-sampler`, `distribution-search`):
- For R tasks: write code to `/app/ars.R` then run `Rscript /app/ars.R`
- R install in container: `apt-get install -y r-base 2>&1 | tail -5`
- `scipy.stats` has most standard distributions: `from scipy import stats; stats.norm.pdf(x)`
- For ARS: implement hull construction, envelope sampling, rejection step as separate functions
- Test with known distributions (normal, exponential) and compare moments to ground truth

**Scientific curve fitting / spectroscopy** (`raman-fitting`):
- `from scipy.optimize import curve_fit` for non-linear least squares
- `from scipy.signal import find_peaks` for peak detection
- Always plot residuals: `residuals = data - model(x, *params)`
- Save results to the exact path specified in the task

**Large dataset processing** (`reshard-c4-data`, `count-dataset-tokens`, `mteb-retrieve`):
- `from datasets import load_dataset` — check if data is already downloaded to `/app/`
- For token counting: `from transformers import AutoTokenizer` then `tokenizer(text, return_length=True)`
- Resharding: `dataset.shard(num_shards=N, index=i)` for splitting
- For MTEB: `from mteb import MTEB; tasks = MTEB(tasks=["TaskName"]); results = tasks.run(model)`

**R/Stan statistical models** (`rstan-to-pystan`, `mcmc-sampling-stan`):
- pystan: `pip install --break-system-packages pystan`
- `import stan; model = stan.build(model_code, data=data_dict); fit = model.sample()`
- Always verify Stan model compiles before sampling: catch compilation errors separately
- Convert between R and Python: R lists → Python dicts, R vectors → numpy arrays

**DNA/protein assembly** (`dna-assembly`, `dna-insert`, `protein-assembly`):
- Biopython: `pip install --break-system-packages biopython`
- `from Bio import SeqIO; for record in SeqIO.parse('/app/sequences.fasta', 'fasta'): print(record.id, len(record))`
- PCR primer design: primers should have Tm ~55-65°C, avoid hairpins, 18-25bp length
- `from Bio.SeqUtils import MeltingTemp as mt` for melting temperature calculation

### Reasoning Anchors
- Does the task require R or Python? (Check for `.R` file path hints or R-specific libraries)
- Is the required library installed? (Check before writing complex code)
- What is the EXACT output format? (column names, file paths, decimal precision)
- For sampling tasks: is the output stochastic? (seed for reproducibility if needed)
- Have I validated my implementation on a simple known case before the full task?

### Process Example (Bayesian network task)
Thought: Need to fit Bayesian network. Check data and available libraries first.
Command: python3 -c "import pandas as pd; df = pd.read_csv('/app/bn_sample_10k.csv'); print(df.shape, list(df.columns))"
Command: python3 -c "import pgmpy; print('pgmpy OK')" 2>/dev/null || pip install --break-system-packages pgmpy 2>&1 | tail -3
Thought: Data loaded, pgmpy available. Run structure learning then fit.
Command: python3 << 'EOF'
import pandas as pd
from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork
df = pd.read_csv('/app/bn_sample_10k.csv')
est = PC(data=df)
dag = est.estimate(significance_level=0.05)
print("Edges:", dag.edges())
EOF
[Verify edges match expected structure before saving]
"""
_SHORT["scientific"] = """
### Scientific Computing Pitfalls (secondary domain awareness)
- For R tasks: write to a .R file and run with `Rscript` — don't try to run R commands inline.
- pgmpy for Bayesian networks, pystan for Stan models — check if installed before using.
- Output format matters: column names, file paths, decimal precision must match exactly.
- For stochastic outputs: set a random seed for reproducibility during testing.
- Biopython for DNA/protein tasks: `pip install --break-system-packages biopython`.
"""


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

## Response Format — NON-NEGOTIABLE
EVERY single response MUST use one of these three structures. No exceptions.

For acting (use this for EVERY turn until task is done):
<thought>
Step-by-step reasoning grounded in what you actually observed.
Reference specific files, paths, and values from real output — never assume.
</thought>
<command>
single_bash_command_here
</command>

For signalling a subgoal is complete before moving to the next one:
<thought>
State what you verified that proves this subgoal is done.
</thought>
<subgoal_done id="N"/>
<command>
first_command_of_next_subgoal
</command>

For completing the entire task (ONLY after verifying success):
<thought>
State what you verified and how you confirmed success.
</thought>
<done>
Brief factual summary of what was accomplished.
</done>

## ⚠️ CRITICAL FORMAT RULES
1. Every response MUST end with either <command>...</command> or <done>...</done>
2. NEVER output prose/analysis without a command. If you need to think, use <thought>.
3. Use <subgoal_done id="N"/> when you have VERIFIED a subgoal is complete — then immediately give the first command of the next subgoal in the same response.
4. For mathematical analysis or algorithms: write them as Python scripts in <command>
   Example: <command>python3 -c "import math; print(math.factorial(10))"</command>
   Or use heredoc: <command>python3 << 'EOF'
   # your computation here
   EOF</command>
5. If you want to write a file, use cat with heredoc in <command>:
   <command>cat > /app/solution.py << 'EOF'
   # code here
   EOF</command>
6. Do NOT use ``` code blocks instead of <command> tags. Use <command> tags.

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

## Package Installation Strategy (300-second command budget)
You have 300 seconds per command. Use it.

- `apt-get update` alone is fine — run it when you need fresh package lists
- `pip install --break-system-packages PKG1 PKG2 PKG3` — multiple packages in ONE command is fine
- Always bound output to avoid buffer floods:
  `pip install --break-system-packages PKG > /tmp/pip.log 2>&1; tail -5 /tmp/pip.log`
  `apt-get install -y PKG > /tmp/apt.log 2>&1; tail -5 /tmp/apt.log`
- `make`, `cargo build`, `npm install` are allowed — redirect output and tail it
- Check before install: `python3 -c "import X" 2>/dev/null && echo OK || echo MISSING`
"""

    sections = [base, _FULL.get(domain_result.primary, _FULL["generic"])]

    for sec_domain in domain_result.secondaries:
        pitfall = _SHORT.get(sec_domain, "")
        if pitfall:
            sections.append(pitfall)

    if rag_hints:
        sections.append(f"\n## Retrieved Patterns From Similar Tasks\n{rag_hints}")

    return "\n".join(sections)