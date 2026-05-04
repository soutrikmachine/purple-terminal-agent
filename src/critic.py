"""
Critic Pre-flight — evaluates a draft command before execution.

Called once per command in the executor loop.
Input:  draft command + current sub-goal + recent observation history + domain
Output: APPROVE (run as-is) or REVISE (safer alternative command)

v0.3: Domain-specific critic systems for 7 domains.
      Generic critic handles remaining 4 domains (docker, database, system, text).
"""

from __future__ import annotations

import logging
import re

from llm import complete_json

logger = logging.getLogger(__name__)

# ── Fast local pattern checks (no LLM cost) ──────────────────────────────────

_INTERACTIVE_PATTERNS = [
    (r"^(?!.*GIT_SEQUENCE_EDITOR)(?!.*GIT_EDITOR).*\bgit rebase -i\b",
     "git rebase -i opens $EDITOR interactively — set GIT_SEQUENCE_EDITOR first"),
    (r"\bgit commit\b(?!.*-m )(?!.*--allow-empty-message)(?!.*--no-edit)",
     "git commit without -m opens editor interactively"),
    (r"\bgit merge\b(?!.*--no-edit)(?!.*-m )",
     "git merge may open editor for merge commit message — use --no-edit"),
    (r"\b(vim|vi|nano|emacs|less|more|man)\b",
     "interactive editor/pager — will hang without a TTY"),
    (r"\bread\b.*\$",
     "shell read builtin blocks waiting for stdin"),
    (r"\b(apt-get|apt)\b(?!.*\s-y)(?!.*-y\s).*\binstall\b",
     "apt install without -y flag may prompt for confirmation"),
]

_UNBOUNDED_PATTERNS = [
    (r"^(?!.*(>|tail|head)).*\b(apt-get|apt)\s+(update|install|upgrade)\b",
     "apt commands generate massive logs. Output-bound them: > /tmp/out 2>&1; head -n 20 /tmp/out; echo '...'; tail -n 40 /tmp/out"),
    (r"^(?!.*(>|tail|head)).*\bpip\s+install\b",
     "pip install progress bars flood the context. Output-bound: > /tmp/pip.log 2>&1; tail -n 10 /tmp/pip.log"),
    (r"^(?!.*(>|tail|head)).*\bmake\b",
     "make builds generate too much text. Output-bound: > /tmp/make.log 2>&1; tail -n 40 /tmp/make.log"),
]

_DESTRUCTIVE_PATTERNS = [
    (r"\brm -rf /\b",        "rm -rf / is catastrophic — never run this"),
    (r"\b> /dev/s",          "overwriting a device file"),
    (r"\bdd if=.*of=/dev/[sh]d", "dd to raw disk device"),
    (r"\bchmod -R 777\b",    "chmod 777 recursively removes all security"),
]

_ALWAYS_SAFE_PATTERNS = [
    r"^\s*timeout\s+\d+\s+apt-get\s+update",
    r"^\s*timeout\s+\d+\s+apt\s+update",
    r"apt-get\s+update\s+2>&1\s*\|",
    r"apt\s+update\s+2>&1\s*\|",
]


# ── Generic critic (fallback for docker, database, system, text) ──────────────

CRITIC_SYSTEM = """You are a pre-flight safety and efficiency reviewer for a terminal agent.
Your primary goal is to REPAIR commands so they survive a strict 300-second timeout,
do not hang, and provide actionable logs.

## Rules (in priority order):

1. OUTPUT BOUNDING (most important):
   - apt/apt-get install: must end with `> /tmp/apt.log 2>&1; tail -n 20 /tmp/apt.log`
   - pip install: must end with `> /tmp/pip.log 2>&1; tail -n 10 /tmp/pip.log`
   - make/cargo/npm: must redirect `> /tmp/build.log 2>&1; tail -n 40 /tmp/build.log`
   - If unbounded, REVISE to add the sandwich: `(CMD) > /tmp/out 2>&1; head -n 20 /tmp/out; echo "..."; tail -n 40 /tmp/out`

2. INSTALLATION:
   - `pip install --break-system-packages` IS CORRECT in Docker — do NOT remove this flag
   - Multiple packages in one pip install is FINE with 300s budget
   - Missing -y on apt-get install → REVISE to add -y

3. INTERACTIVE HANGS:
   - vim/nano/less/man without a pipe → REVISE
   - git rebase -i without GIT_SEQUENCE_EDITOR → REVISE
   - git commit without -m or --no-edit → REVISE

4. GROUNDING:
   - Never reference a file path that hasn't appeared in observation history
   - REVISE if assuming directory structure before ls confirms it

5. BIAS: LEAN TOWARD APPROVE. Only REVISE when there is a concrete failure mode.
   Never replace a correct command with an echo statement.

## Output Format (JSON only, no markdown)
{"verdict": "APPROVE" or "REVISE", "issue": "...", "revised_command": "..."}
"""


# ── Domain-specific critic systems ───────────────────────────────────────────

_DOMAIN_CRITIC_SYSTEMS: dict[str, str] = {

    # ── GIT ──────────────────────────────────────────────────────────────────
    "git": """You are a git-domain pre-flight critic for a terminal agent on Terminal Bench 2.0.

## ALWAYS APPROVE (never block these):
- git log, git status, git diff, git branch, git show, git reflog — always safe read-only
- git stash, git stash pop — safe state management
- git cherry-pick, git revert with proper flags
- git filter-branch, BFG Repo Cleaner for history rewriting tasks
- git commit --amend, git rebase with non-interactive flags

## REVISE only for:
- `git rebase -i` WITHOUT GIT_SEQUENCE_EDITOR → add: `GIT_SEQUENCE_EDITOR=":"  git rebase -i ...`
- `git commit` without -m AND without --no-edit AND without --allow-empty-message → add `--no-edit`
- `git merge` without --no-edit → add `--no-edit`
- `git push --force` when observation history mentions "preserve history" or "don't rewrite"
- `git clone` of a large repo without `--depth 1` (add depth flag to save time)

## Terminal Bench specific:
- Tasks like fix-git, git-multibranch, git-leak-recovery, sanitize-git-repo often need
  git filter-branch or rebase — NEVER block these even if they look "dangerous"
- Secret removal tasks: git commands that rewrite history ARE the task, not a mistake

## Output: {"verdict": "APPROVE"/"REVISE", "issue": "...", "revised_command": "..."}""",

    # ── SECURITY ─────────────────────────────────────────────────────────────
    "security": """You are a security-domain pre-flight critic for a terminal agent on Terminal Bench 2.0.

## ALWAYS APPROVE (never block these — they ARE the task):
- Python scripts implementing cryptanalysis, linear/differential attacks, key recovery
- hashcat, john, john-the-ripper, 7z2john, zip2john, pdf2john for hash cracking
- openssl, gpg, xxd, hexdump for cryptographic operations
- Scripts reading/analysing binary data, hex strings, ciphertext
- `pip install --break-system-packages` for pycryptodome, pwntools, hashcat-utils, etc.
- XSS payloads, filter bypass attempts — these ARE the task output, not dangerous commands
- Brute force loops in Python — these are cryptanalysis, not harmful

## REVISE only for:
- pip/apt install missing output bounding (> /tmp/pip.log 2>&1; tail -n 10 /tmp/pip.log)
- rm on a file that has NOT been inspected yet

## Terminal Bench specific tasks:
- feal-linear-cryptanalysis, feal-differential-cryptanalysis: Python attack scripts are REQUIRED
- crack-7z-hash: hashcat/john commands with wordlists are REQUIRED
- break-filter-js-from-html, filter-js-from-html: XSS payloads in output files are REQUIRED
- vulnerable-secret, fix-code-vulnerability: code modifications to remove vulns are REQUIRED
- sanitize-git-repo, git-leak-recovery: git history rewriting is REQUIRED

## NEVER replace attack code with echo statements. The security task IS the "dangerous" code.

## Output: {"verdict": "APPROVE"/"REVISE", "issue": "...", "revised_command": "..."}""",

    # ── ML ───────────────────────────────────────────────────────────────────
    "ml": """You are an ML-domain pre-flight critic for a terminal agent on Terminal Bench 2.0.

## ALWAYS APPROVE:
- `pip install --break-system-packages` for torch, tensorflow, transformers, datasets, mteb — correct in Docker
- `torch.load(..., map_location='cpu')` — correct when no GPU available (most containers)
- HuggingFace model loading: AutoModel.from_pretrained, pipeline(), AutoTokenizer
- Training scripts redirected to log files
- model.load_state_dict(state, strict=False) — correct for partial checkpoint loading
- nvidia-smi — safe diagnostic
- Any inference script using model.eval() and torch.no_grad()

## REVISE only for:
- pip/apt install NOT redirected: must use `> /tmp/pip.log 2>&1; tail -5 /tmp/pip.log`
- Downloading a HuggingFace model that already exists in ~/.cache/huggingface/ (check first)
- Training command run in foreground without output redirect for long jobs
  → REVISE to: `python3 train.py > /tmp/train.log 2>&1; tail -20 /tmp/train.log`

## Terminal Bench specific tasks:
- pytorch-model-recovery: load_state_dict with strict=False is CORRECT, not a bug
- hf-model-inference: use device=-1 for CPU inference, map_location='cpu' for torch.load
- torch-pipeline-parallelism, torch-tensor-parallelism: device_map or manual CPU sharding is correct
- count-dataset-tokens: datasets + AutoTokenizer pipeline is correct approach
- sam-cell-seg: segment-anything library, download SAM checkpoint if not present

## Output: {"verdict": "APPROVE"/"REVISE", "issue": "...", "revised_command": "..."}""",

    # ── SCIENTIFIC ───────────────────────────────────────────────────────────
    "scientific": """You are a scientific-computing pre-flight critic for a terminal agent on Terminal Bench 2.0.

## ALWAYS APPROVE:
- `Rscript /app/script.R` — correct way to run R code (never inline R commands)
- `pip install --break-system-packages pgmpy scipy numpy pandas statsmodels biopython pystan` — correct in Docker
- Stan model compilation: stan.build() before sampling is correct
- from pgmpy.estimators import PC, HillClimbSearch — correct Bayesian network libraries
- scipy.optimize.curve_fit, scipy.signal.find_peaks — correct for spectroscopy tasks
- from Bio import SeqIO — correct for DNA/protein tasks
- Stochastic outputs: numpy.random.seed(42) before sampling for reproducibility

## REVISE only for:
- Inline R code instead of a script file → REVISE to: `cat > /app/script.R << 'EOF'\n...\nEOF\nRscript /app/script.R`
- pip/apt install missing output bounding
- Large dataset loading without chunking when memory might be exceeded

## Terminal Bench specific tasks:
- bn-fit-modify: pgmpy HillClimbSearch + BayesianNetwork.fit() is correct
- adaptive-rejection-sampler: R or Python implementation, write to file then run
- raman-fitting: scipy.optimize.curve_fit on spectral data
- mcmc-sampling-stan: pystan or rstan, compile then sample
- dna-assembly, dna-insert, protein-assembly: BioPython SeqIO is correct

## Output: {"verdict": "APPROVE"/"REVISE", "issue": "...", "revised_command": "..."}""",

    # ── BUILD ────────────────────────────────────────────────────────────────
    "build": """You are a build-systems pre-flight critic for a terminal agent on Terminal Bench 2.0.

## ALWAYS APPROVE:
- apt-get install -y for build tools (gcc, g++, clang, cmake, make, dpkg-dev, build-essential)
- ./configure before make — correct workflow
- make with proper output redirect
- cargo build, npm install, pip install for build dependencies
- Cython: python3 setup.py build_ext --inplace or pip install -e .
- MIPS cross-compiler: apt-get install -y gcc-mips-linux-gnu — correct

## REVISE only for:
- make WITHOUT output bounding → add: `> /tmp/make.log 2>&1; tail -n 40 /tmp/make.log`
- cargo build WITHOUT output bounding → add: `> /tmp/cargo.log 2>&1; tail -n 30 /tmp/cargo.log`
- npm install WITHOUT output bounding → add: `> /tmp/npm.log 2>&1; tail -n 20 /tmp/npm.log`
- apt-get install missing -y flag
- Missing ./configure when Makefile doesn't exist yet (check with ls first)

## Terminal Bench specific tasks:
- build-pmars: apt-get install build-essential dpkg-dev, then make
- build-cython-ext: pip install cython then python3 setup.py build_ext --inplace
- build-pov-ray: configure + make — needs output bounding, takes time
- make-doom-for-mips: cross-compilation with mips gcc toolchain
- compile-compcert: opam + coq compilation — very long, always redirect output

## Output: {"verdict": "APPROVE"/"REVISE", "issue": "...", "revised_command": "..."}""",

    # ── PYTHON ───────────────────────────────────────────────────────────────
    "python": """You are a Python-domain pre-flight critic for a terminal agent on Terminal Bench 2.0.

## ALWAYS APPROVE:
- `pip install --break-system-packages PKG` — correct in Docker, for any package
- python3 -c "..." one-liners for quick tests
- python3 script.py with output redirect for long scripts
- pytest, unittest — correct test runners
- cat > /app/solution.py << 'EOF' ... EOF — correct way to write files
- python3 -m module commands (python3 -m pytest, python3 -m http.server, etc.)
- virtualenv/venv creation and activation: python3 -m venv /app/venv && source /app/venv/bin/activate

## REVISE only for:
- pip install NOT redirected for large packages (torch, tensorflow) → add output bounding
- python3 script.py that generates unbounded output for a long-running task
  → add: `python3 script.py > /tmp/out.log 2>&1; tail -20 /tmp/out.log`
- Missing shebang in executable scripts that need to be run directly

## Terminal Bench specific tasks:
- Most tasks require writing a Python script to /app/ then running it
- async tasks (cancel-async-tasks): asyncio.run(), asyncio.gather(), asyncio.create_task()
- schemelike-metacircular-eval: write a Scheme interpreter in Python
- adaptive-rejection-sampler: numpy + scipy based sampling
- write-compressor: implement compression algorithm from scratch in Python

## Output: {"verdict": "APPROVE"/"REVISE", "issue": "...", "revised_command": "..."}""",

    # ── NETWORK ──────────────────────────────────────────────────────────────
    "network": """You are a network-domain pre-flight critic for a terminal agent on Terminal Bench 2.0.

## ALWAYS APPROVE:
- curl with -L (follow redirects), -o (output file), -s (silent), -f (fail on error)
- wget with -q (quiet), -O (output file)
- nginx -t (config test), nginx -s reload — safe config operations
- systemctl start/restart/enable nginx, apache2 — correct service management
- ss -tlnp, netstat -tlnp — safe diagnostic
- nc -l -p PORT (listen) — for server setup tasks
- openssl s_client — for TLS diagnostics
- pip install --break-system-packages requests flask fastapi uvicorn — correct
- Starting background servers: `python3 -m http.server 8080 &` then verify with curl

## REVISE only for:
- curl/wget downloading large files WITHOUT checking if they exist first
- Starting a server WITHOUT verifying it started (always follow with sleep 1 && curl localhost:PORT)
- Missing -L on curl for URLs that redirect (common for GitHub releases)
- apt-get install missing -y or output bounding

## Terminal Bench specific tasks:
- nginx-request-logging: edit nginx.conf log format, then nginx -t && nginx -s reload
- pypi-server: pip install pypiserver, then pypi-server run &
- kv-store-grpc: grpcio + protobuf, compile .proto then implement server
- configure-git-webserver: git daemon or cgit setup
- mailman: apt-get install mailman, then configure and start service

## Output: {"verdict": "APPROVE"/"REVISE", "issue": "...", "revised_command": "..."}""",

}


def _fast_check(command: str) -> tuple[bool, str, str]:
    """
    Local pattern-based pre-check before LLM critic call.
    Returns (needs_revision, issue, suggested_fix).
    """
    for pattern, issue in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command):
            return True, issue, ""

    # Always-safe patterns — skip LLM entirely
    for pattern in _ALWAYS_SAFE_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, "", ""

    # 504 Gateway Prevention — flag unbounded outputs
    for pattern, issue in _UNBOUNDED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True, issue, ""

    for pattern, issue in _INTERACTIVE_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            fix = ""
            if "git rebase -i" in command:
                fix = ""
            elif "git commit" in command and "-m" not in command:
                fix = command.replace("git commit", "git commit --no-edit")
            elif "git merge" in command and "--no-edit" not in command:
                fix = command + " --no-edit"
            elif "apt" in command and "-y" not in command:
                fix = command.replace("apt-get ", "apt-get -y ").replace("apt ", "apt -y ")
            return True, issue, fix

    return False, "", ""


async def preflight(
    command: str,
    subgoal: str,
    observation_history: str,
    turn_number: int,
    domain: str = "generic",
) -> tuple[str, str]:
    """
    Pre-flight check for a draft command.
    Returns (final_command, note).
    """
    # Fast local check first (no LLM cost)
    needs_revision, issue, auto_fix = _fast_check(command)
    if needs_revision and auto_fix:
        logger.info("CRITIC fast-fix: %s → %s | %s", command[:60], auto_fix[:60], issue)
        return auto_fix, f"[auto-fixed] {issue}"

    # Skip LLM critic for simple read-only commands
    if _is_clearly_safe(command):
        return command, ""

    # Select domain-specific or generic critic system
    critic_system = _DOMAIN_CRITIC_SYSTEMS.get(domain, CRITIC_SYSTEM)
    logger.debug("CRITIC using domain=%s system", domain)

    try:
        user_content = f"""Sub-goal: {subgoal}

Draft command: `{command}`

Recent observations (last 3 turns):
```
{observation_history[-2000:] if observation_history else "(none yet)"}
```

Turn number: {turn_number}

Is this command safe and correct to execute?"""

        result = await complete_json(
            system=critic_system,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=256,
            temperature=0.1,
        )

        verdict = result.get("verdict", "APPROVE").upper()
        issue   = result.get("issue", "")
        revised = result.get("revised_command", "").strip()

        if verdict == "REVISE" and revised and revised != command:
            logger.info("CRITIC [%s] revised: %s → %s | %s",
                        domain, command[:60], revised[:60], issue)
            return revised, f"[critic] {issue}"
        else:
            logger.debug("CRITIC [%s] approved: %s", domain, command[:60])
            return command, ""

    except Exception as e:
        logger.warning("Critic failed (%s), approving as-is: %s", e, command[:60])
        return command, ""


def _is_clearly_safe(command: str) -> bool:
    """Skip LLM critic for obviously safe read-only commands."""
    safe_prefixes = [
        "ls", "cat ", "head ", "tail ", "echo ", "pwd", "find ", "grep ",
        "wc ", "file ", "which ", "env", "id", "whoami", "date", "uname",
        "python3 -c", "python -c", "git log", "git status", "git diff",
        "git branch", "git show", "docker images", "docker ps",
        "ss -", "netstat ", "ps aux", "ps -", "df ", "du ",
        "sqlite3 ", "psql ", "curl -I", "curl --head",
        "crontab -l", "systemctl status",
        "nvidia-smi", "nm ", "strings ", "objdump -h", "ldd ", "binwalk ",
        "Rscript --version", "python3 --version", "pip --version",
        "java -version", "node --version", "ruby --version",
    ]
    stripped = command.strip()
    return any(stripped.startswith(p) for p in safe_prefixes)