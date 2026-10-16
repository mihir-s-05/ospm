"""Execution environment for agent actions (mini-swe-agent style: every action is an independent subprocess).

LocalEnvironment is a *best-effort* sandbox, not a container:
* the child environment is rebuilt from an allow-list - API keys, tokens and the user's HOME, gh config, SSH and
  git credential helpers are not visible, so agent commands cannot act with the user's credentials;
* commands run in the episode workspace with a timeout that kills the whole process tree;
* a small deny-list blocks pushes, the gh CLI and obviously destructive commands.
It does not block network access or file access outside the workspace; run inside a VM/container for untrusted
repositories (the environment class is the only thing to swap, exactly as in mini-swe-agent).
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

_DENY = re.compile(
    # pushes, the gh CLI, and git network commands that could pull in refs from after the replay time
    r"(^|[;&|`(]\s*)(git\s+(-C\s+\S+\s+)?(push|fetch|pull|remote|ls-remote|clone|submodule)\b|gh\s|sudo\s|"
    r"shutdown|reboot|mkfs|format\s+[a-z]:|"
    r"rm\s+-[a-z]*r[a-z]*f?\s+(/|~|\$HOME|[a-z]:[\\/])(\s|$)|:\(\)\s*\{)",
    re.IGNORECASE,
)
_PASS_ENV = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TZ", "LANG", "LC_ALL",
             "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS")


def find_bash() -> str:
    env = os.environ.get("OSPM_BASH")
    if env:
        return env
    if sys.platform == "win32":
        for cand in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if os.path.exists(cand):
                return cand
    found = shutil.which("bash")
    if found and not found.lower().endswith(r"system32\bash.exe"):  # that one is the WSL launcher
        return found
    raise RuntimeError("bash not found (install Git for Windows or set OSPM_BASH)")


def _msys_path(p: str) -> str:
    """`D:\\x\\y` -> `/d/x/y` (a PATH entry for Git for Windows' bash); POSIX paths are unchanged."""
    m = re.match(r"^([A-Za-z]):[\\/]?(.*)$", p)
    return f"/{m[1].lower()}/{m[2].replace(chr(92), '/')}" if m else p


@dataclass
class ExecResult:
    output: str
    returncode: int
    timed_out: bool = False
    blocked: bool = False


@dataclass
class LocalEnvironment:
    cwd: str
    env: dict[str, str] = field(default_factory=dict)  # extra variables (history db, as-of time, ...)
    bin_dirs: list[str] = field(default_factory=list)  # prepended to PATH (helper commands)
    timeout: int = 120
    bash: str = field(default_factory=find_bash)
    home: str = field(default_factory=lambda: tempfile.mkdtemp(prefix="ospm-home-"))

    def __post_init__(self) -> None:
        Path(self.home).mkdir(parents=True, exist_ok=True)
        self._git_global = str(Path(self.home) / ".gitconfig-empty")
        Path(self._git_global).touch()
        Path(self.home, "gh").mkdir(exist_ok=True)
        self._bash_env = str(Path(self.home) / ".bash_env")
        dirs = ":".join(_msys_path(d) for d in self.bin_dirs)
        Path(self._bash_env).write_text(f'PATH="{dirs}:$PATH"\n', encoding="utf-8", newline="\n")

    def child_env(self) -> dict[str, str]:
        env = {k: os.environ[k] for k in _PASS_ENV if k in os.environ}
        path = os.environ.get("PATH", "")
        env["PATH"] = os.pathsep.join([*self.bin_dirs, path])
        env.update({
            "HOME": self.home, "USERPROFILE": self.home,
            "GH_CONFIG_DIR": str(Path(self.home) / "gh"),  # unauthenticated gh even if someone calls it
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": self._git_global, "GIT_TERMINAL_PROMPT": "0",
            "GIT_SSH_COMMAND": "false", "GIT_PAGER": "cat", "PAGER": "cat",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PIP_NO_INPUT": "1",
            "TERM": "dumb", "NO_COLOR": "1", "CLICOLOR": "0",
        })
        if self.bin_dirs:
            # Git for Windows' bash puts /mingw64/bin:/usr/bin ahead of the inherited PATH; re-prepend the helper
            # dirs so their commands (the grep wrapper) win over the system ones
            env["BASH_ENV"] = self._bash_env
        env.update(self.env)
        return env

    def execute(self, command: str, timeout: int | None = None) -> ExecResult:
        if _DENY.search(command):
            return ExecResult("Command blocked by the sandbox policy (pushes, gh CLI, sudo and destructive "
                              "commands are not allowed). Use read-only investigation commands.", 126, blocked=True)
        timeout = timeout or self.timeout
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen([self.bash, "-c", command], cwd=self.cwd, env=self.child_env(),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
        try:
            out, _ = proc.communicate(timeout=timeout)
            return ExecResult(out.decode("utf-8", "replace"), proc.returncode)
        except subprocess.TimeoutExpired:
            self._kill(proc)
            out = b""
            try:
                out, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return ExecResult(out.decode("utf-8", "replace") + f"\n[command timed out after {timeout}s and was killed]",
                              124, timed_out=True)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=30)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            proc.kill()

    def cleanup(self) -> None:
        shutil.rmtree(self.home, ignore_errors=True)


def format_observation(res: ExecResult, max_chars: int = 10000) -> str:
    """mini-swe-agent-like observation with head/tail truncation of long outputs."""
    out = res.output
    if len(out) > max_chars:
        head, tail = out[: max_chars // 2], out[-max_chars // 2:]
        out = (f"{head}\n... [{len(res.output) - max_chars} characters elided - the output was too long; use "
               f"head/tail/grep or more specific commands] ...\n{tail}")
    return f"<returncode>{res.returncode}</returncode>\n<output>\n{out}</output>"


def extract_submission(output: str) -> str | None:
    """mini-swe-agent v2 protocol: if the first non-empty output line is the marker, the rest is the submission."""
    lines = output.lstrip().splitlines()
    if lines and lines[0].strip() == SUBMIT_MARKER:
        return "\n".join(lines[1:]).strip()
    return None
