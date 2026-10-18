"""The agent's tools: Pi's default four (bash, read, edit, write) + submit, run against one workspace.

Mechanics shared by Claude Code / Pi / OpenCode / Hermes:
  * read returns line-numbered text with offset/limit;
  * edit is exact string replacement, must match exactly once unless replace_all, and requires
    a prior read of that file in this episode;
  * every failure comes back as a tool result (never an exception), so the model can recover;
  * long outputs are head/tail truncated with a marker that says how to see more.
bash runs inside bubblewrap: no network, read-only /usr, only the workspace is writable, with
CPU/memory/file-size/process limits and a wall-clock timeout.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

MAX_RESULT_CHARS = 6000          # ~1.5k tokens: sized for an 8k-context model, not harness defaults
READ_DEFAULT_LIMIT = 200
BASH_TIMEOUT_S = 10

TOOL_SPECS = [
    ("bash", "command", "Run a shell command in /work (no network)."),
    ("read", "path, offset?, limit?", "Read a file with line numbers (offset is 1-based)."),
    ("edit", "path, old_string, new_string, replace_all?", "Replace exact text in a file you have read."),
    ("write", "path, content", "Create or overwrite a file."),
    ("submit", "answer", "Finish with the final answer."),
]


def truncate(text: str, limit: int = MAX_RESULT_CHARS, hint: str = "") -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 3], text[-limit // 3:]
    omitted = text[len(head): len(text) - len(tail)]
    n_lines = omitted.count("\n")
    return f"{head}\n[... truncated {len(omitted)} chars / {n_lines} lines{hint} ...]\n{tail}"


def bwrap_cmd(workdir: str, command: str, timeout: int) -> list[str]:
    limits = "ulimit -v 2000000; ulimit -t %d; ulimit -f 20000; ulimit -u 256;" % timeout
    return ["timeout", "-k", "1", str(timeout), "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/sbin", "/sbin", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--bind", workdir, "/work", "--chdir", "/work", "--clearenv",
            "--setenv", "HOME", "/work", "--setenv", "PATH", "/usr/bin", "--setenv", "LANG", "C.UTF-8",
            # no .pyc: an edit within the same second and of the same size would reuse stale bytecode
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            *(["--ro-bind", "/etc/abrt", "/etc/abrt"] if os.path.isdir("/etc/abrt") else []),
            "--", "bash", "-c", limits + command]


@dataclass
class Workspace:
    """A throwaway directory on tmpfs holding one episode's files."""
    files: dict[str, str]
    root: str = ""
    read_paths: set = field(default_factory=set)
    submitted: str | None = None

    def __post_init__(self):
        base = "/dev/shm/tiny-agent-ws"
        os.makedirs(base, exist_ok=True)
        self.root = tempfile.mkdtemp(dir=base)
        for rel, content in self.files.items():
            p = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(content)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- path handling: tool paths are relative to /work (absolute /work/... also accepted)
    def _resolve(self, path: str) -> str | None:
        if path.startswith("/work/"):
            path = path[len("/work/"):]
        elif path == "/work":
            path = "."
        full = os.path.realpath(os.path.join(self.root, path))
        if full != self.root and not full.startswith(self.root + os.sep):
            return None
        return full

    def call(self, name: str, args: dict) -> str:
        try:
            fn = getattr(self, f"tool_{name}", None)
            if fn is None:
                return f"Error: unknown tool '{name}'. Available: bash, read, edit, write, submit."
            return fn(**args)
        except (TypeError, ValueError, AttributeError, OverflowError, OSError) as e:   # offset="ten", path=5, a dir
            return f"Error: bad arguments for {name}: {e}"

    def tool_bash(self, command: str, timeout: int = BASH_TIMEOUT_S) -> str:
        timeout = max(1, min(int(timeout), 30))
        try:
            r = subprocess.run(bwrap_cmd(self.root, command, timeout), capture_output=True, timeout=timeout + 5)
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        out = (r.stdout + r.stderr).decode("utf-8", "replace")
        if r.returncode == 124:
            out += f"\n[timed out after {timeout}s]"
        elif r.returncode:
            out += f"\n[exit code {r.returncode}]"
        return truncate(out.strip() or "(no output)", hint=", narrow the command (e.g. | head, grep)")

    def tool_read(self, path: str, offset: int = 1, limit: int = READ_DEFAULT_LIMIT) -> str:
        full = self._resolve(path)
        if full is None:
            return "Error: path is outside /work"
        if os.path.isdir(full):
            return f"Error: {path} is a directory. Use bash with ls."
        if not os.path.exists(full):
            return f"Error: file not found: {path}"
        with open(full, errors="replace") as f:
            lines = f.read().split("\n")
        self.read_paths.add(full)
        offset, limit = max(1, int(offset)), max(1, int(limit))
        chunk = lines[offset - 1: offset - 1 + limit]
        body = "\n".join(f"{offset + i:6d}\t{l}" for i, l in enumerate(chunk))
        rest = len(lines) - (offset - 1 + len(chunk))
        if rest > 0:
            body += f"\n[{rest} more lines, use offset={offset + len(chunk)}]"
        return truncate(body or "(empty file)", hint=", use a smaller limit")

    def tool_edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        full = self._resolve(path)
        if full is None:
            return "Error: path is outside /work"
        if not os.path.exists(full):
            return f"Error: file not found: {path}"
        if full not in self.read_paths:
            return f"Error: read {path} before editing it"
        text = open(full).read()
        n = text.count(old_string)
        if n == 0:
            return "Error: old_string not found in file"
        if n > 1 and not replace_all:
            return f"Error: old_string appears {n} times; add context to make it unique or set replace_all"
        with open(full, "w") as f:
            f.write(text.replace(old_string, new_string))
        return f"Edited {path} ({n} replacement{'s' if n > 1 else ''})"

    def tool_write(self, path: str, content: str) -> str:
        full = self._resolve(path)
        if full is None:
            return "Error: path is outside /work"
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        self.read_paths.add(full)
        return f"Wrote {path} ({content.count(chr(10)) + 1} lines)"

    def tool_submit(self, answer: str) -> str:
        self.submitted = str(answer)
        return "Submitted."
