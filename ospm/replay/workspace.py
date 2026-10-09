"""Replay workspaces: git worktrees at historical commits, sandbox helper commands, and local CI.

Storage-conscious by design: worktrees are created per episode from the bare mirror and removed afterwards;
one shared virtualenv (the project's runtime dependencies only) serves every local CI run.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Iterator

from ..harness.env import LocalEnvironment
from .ghstack import base_of as ghstack_base
from .ghstack import is_ghstack
from .store import HistoryStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# runtime deps of pytest itself + a few test-suite extras; deliberately small
CI_REQUIREMENTS = ["iniconfig", "packaging", "pluggy>=1.5,<2", "pygments>=2.7.2", "colorama", "attrs",
                   "hypothesis", "xmlschema"]
VERSION_FILE = 'version = __version__ = "9.9.9.dev0"\nversion_tuple = __version_tuple__ = (9, 9, 9, "dev0")\n'


# In a blobless mirror, touching a missing object makes git silently fetch it from GitHub (a lazy fetch), which
# for a force-pushed-away commit pulls tens of MB of trees per object. Metadata queries never fetch: a missing
# commit is reported as missing and the task is skipped. Only workspace checkouts may fetch file contents.
_NO_LAZY = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}


def _git(git_dir: Path, *args: str, check: bool = True, lazy: bool = False) -> str:
    out = subprocess.run(["git", "--git-dir", str(git_dir), *args], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=None if lazy else _NO_LAZY)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()[:300]}")
    return out.stdout.strip()


def _rmtree(path: Path) -> None:
    """rmtree that also removes read-only files (git writes its pack files read-only, which Windows refuses to
    delete): leftovers would otherwise pile up at ~5 MB per episode."""
    import stat

    def force(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=force)
    else:
        shutil.rmtree(path, onerror=force)


def _posix(p: Path | str) -> str:
    return Path(p).resolve().as_posix()


SKILL_SCRIPT = r'''#!/usr/bin/env bash
d="$(dirname "$0")/skills"
if [ $# -eq 0 ]; then
  if ls "$d"/*.md >/dev/null 2>&1; then
    for f in "$d"/*.md; do head -n 1 "$f"; done
  else
    echo "no skills available"
  fi
  exit 0
fi
case "$1" in
  *[!a-z0-9-]*|-*|"") ok=0 ;;
  *) ok=1 ;;
esac
if [ "$ok" = 1 ] && [ -f "$d/$1.md" ]; then
  cat "$d/$1.md"
  exit 0
fi
echo "unknown skill: $1" >&2
echo "available skills: $(ls "$d" 2>/dev/null | sed -n 's/\.md$//p' | tr '\n' ' ')" >&2
exit 1
'''


def write_skills(bin_dir: Path, skills) -> None:
    """`skill` command: no args lists the skills, `skill <name>` prints one (a header line + the body)."""
    sd = bin_dir / "skills"
    sd.mkdir(parents=True, exist_ok=True)
    for s in skills:
        desc = " ".join(s.description.split())
        (sd / f"{s.name}.md").write_text(f"{s.name}: {desc}\n{s.body.strip()}\n", encoding="utf-8", newline="\n")
    (bin_dir / "skill").write_text(SKILL_SCRIPT, encoding="utf-8", newline="\n")


class ReplayRepo:
    """A mirrored repository: history DB + bare git clone + worktree pool + local CI venv."""

    def __init__(self, data_dir: str | Path):
        self.data = Path(data_dir).resolve()
        self.git_dir = self.data / "repo.git"
        self.db_path = self.data / "history.db"
        self.wt_root = self.data / "wt"
        self.venv = self.data / "venv"
        self._lock = threading.Lock()
        if not self.git_dir.exists() or not self.db_path.exists():
            raise FileNotFoundError(f"{self.data} is not a mirror (run `ospm mirror` first)")
        self.store = HistoryStore(self.db_path, readonly=True)

    # ------------------------------------------------------------------ commits
    def commit_before(self, ts: str, branch: str = "main") -> str:
        # --first-parent: a PR merged later carries commits with *earlier* dates; only the branch's own
        # first-parent chain says what was actually on the branch at `ts`
        sha = _git(self.git_dir, "rev-list", "-1", "--first-parent", f"--before={ts}", f"refs/heads/{branch}")
        if not sha:
            raise RuntimeError(f"no commit on {branch} before {ts}")
        return sha

    def has_commit(self, sha: str) -> bool:
        return bool(sha) and subprocess.run(["git", "--git-dir", str(self.git_dir), "cat-file", "-e", f"{sha}^{{commit}}"],
                                            capture_output=True, env=_NO_LAZY).returncode == 0

    def committed_after(self, sha: str, as_of: str) -> bool:
        """The commit was made after `as_of`, so it cannot be what was under review then.

        GitHub reports some reviews (mostly approvals) against a head that was pushed later - e.g. a "merge main"
        commit made days after the approval - and such a commit would show the agent the future.
        """
        from datetime import datetime

        made = datetime.fromisoformat(_git(self.git_dir, "log", "-1", "--format=%cI", sha))
        return made > datetime.fromisoformat(as_of.replace("Z", "+00:00"))

    def merge_base(self, a: str, b: str) -> str:
        return _git(self.git_dir, "merge-base", a, b, check=False)

    def base_at(self, sha: str, as_of: str, branch: str = "main") -> str:
        """Fork point of a PR commit from the target branch *as it was* at `as_of` (later merges excluded)."""
        if is_ghstack(branch):  # synthetic per-PR base branch, long deleted: read the base off the commits
            return ghstack_base(lambda *a: _git(self.git_dir, *a, check=False), sha)
        try:
            return self.merge_base(sha, self.commit_before(as_of, branch))
        except RuntimeError:
            return ""  # target branch no longer exists (e.g. a deleted release branch): callers skip the PR

    # ------------------------------------------------------------------ workspaces
    @property
    def partial(self) -> bool:
        """The mirror is a blobless (partial) clone: file contents are fetched lazily and cached in it."""
        if not hasattr(self, "_partial"):
            # blobless clones are marked by a promisor remote (extensions.partialClone is not always written)
            self._partial = (_git(self.git_dir, "config", "--get", "remote.origin.promisor", check=False) == "true"
                             or bool(_git(self.git_dir, "config", "--get", "extensions.partialClone", check=False)))
            self.promisor_url = _git(self.git_dir, "config", "--get", "remote.origin.url", check=False)
        return self._partial

    def _missing(self, commit: str) -> int:
        """Number of objects of `commit`'s tree not present locally (no network access)."""
        out = subprocess.run(["git", "--git-dir", str(self.git_dir), "rev-list", "--objects", "--no-walk",
                              "--missing=print", commit], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", env=_NO_LAZY).stdout
        return sum(1 for line in out.splitlines() if line.startswith("?"))

    def prefetch(self, sha: str, base: str = "") -> None:
        """Batch-fetch the blobs of `sha`'s tree (and of the files changed since `base`) into the mirror once, so
        workspaces cloned with --shared can check them out and later replays need no network."""
        if not any(self._missing(c) for c in (sha, base) if c):
            return  # everything already cached in the mirror
        scratch = self.wt_root / f"prefetch-{uuid.uuid4().hex[:8]}"
        index = scratch.with_suffix(".index")
        try:
            env = dict(os.environ, GIT_INDEX_FILE=str(index))
            scratch.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "--git-dir", str(self.git_dir), "--work-tree", str(scratch), "read-tree",
                            "--reset", "-u", sha], env=env, capture_output=True, check=False)
            if base:
                subprocess.run(["git", "--git-dir", str(self.git_dir), "diff", "--stat", f"{base}..{sha}"],
                               capture_output=True, check=False)
        finally:
            _rmtree(scratch)
            index.unlink(missing_ok=True)

    @contextlib.contextmanager
    def worktree(self, sha: str, as_of: str | None = None, branch: str = "main", base: str = "") -> Iterator[Path]:
        """A checkout of `sha` that cannot see the future.

        A `git clone --shared` (objects via alternates: no extra object storage) whose only refs are the detached
        HEAD and `main` as it was at `as_of`. Later commits stay unnamed, so `git log main` shows exactly the
        history that existed at the event time.
        """
        path = self.wt_root / uuid.uuid4().hex[:10]
        self.wt_root.mkdir(parents=True, exist_ok=True)
        try:
            if self.partial:
                self.prefetch(sha, base)
            subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(self.git_dir.resolve()),
                            str(path)], check=True, capture_output=True)
            g = path / ".git"
            _git(g, "--work-tree", str(path), "checkout", "--quiet", "--detach", sha, lazy=True)
            # one transaction: a mirror can hold tens of thousands of refs (PyTorch: ~17k branches and PR heads),
            # and one `update-ref -d` process per ref takes many minutes on Windows
            for line in _git(g, "for-each-ref", "--format=%(refname) %(symref)").splitlines():
                ref, _, target = line.partition(" ")
                if target:  # a symref (refs/remotes/origin/HEAD) would make the batch delete fail as a whole
                    _git(g, "symbolic-ref", "--delete", ref, check=False)
            refs = _git(g, "for-each-ref", "--format=%(refname)").split()
            if refs:
                # bytes, not text: on Windows text mode would turn "\n" into "\r\n" and fail every ref name
                subprocess.run(["git", "--git-dir", str(g), "update-ref", "--stdin"], input="".join(
                    f"delete {r}\n" for r in refs).encode(), capture_output=True, env=_NO_LAZY, check=False)
            for ref in _git(g, "for-each-ref", "--format=%(refname)").split():  # whatever the batch left
                _git(g, "update-ref", "-d", ref, check=False)
            _git(g, "remote", "remove", "origin", check=False)
            if self.partial and self.promisor_url:
                # older blobs an agent touches (git show/log -p) are fetched on demand by hash; explicit
                # fetch/pull/remote commands are blocked by the sandbox, so no future refs can be pulled in
                _git(g, "remote", "add", "origin", self.promisor_url, check=False)
                _git(g, "config", "remote.origin.promisor", "true", check=False)
                _git(g, "config", "remote.origin.partialclonefilter", "blob:none", check=False)
                _git(g, "config", "core.repositoryformatversion", "1", check=False)
                _git(g, "config", "extensions.partialClone", "origin", check=False)
            if as_of:
                main_then = self.commit_before(as_of, branch)
                _git(g, "update-ref", f"refs/heads/{branch}", main_then)
            vf = path / "src" / "_pytest" / "_version.py"
            if vf.parent.exists() and not vf.exists():
                vf.write_text(VERSION_FILE, encoding="utf-8")  # normally generated by setuptools-scm
            yield path
        finally:
            _rmtree(path)
            _rmtree(path.parent / f"{path.name}-bin")

    def clean(self) -> None:
        _rmtree(self.wt_root)

    # ------------------------------------------------------------------ local CI
    @property
    def venv_python(self) -> Path:
        return self.venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def ensure_venv(self, log=print) -> Path:
        if self.venv_python.exists():
            return self.venv_python
        log(f"[ci] creating {self.venv} ({', '.join(CI_REQUIREMENTS)})")
        subprocess.run([sys.executable, "-m", "venv", str(self.venv)], check=True)
        subprocess.run([str(self.venv_python), "-m", "pip", "install", "-q", "--no-cache-dir",
                        "--disable-pip-version-check", *CI_REQUIREMENTS], check=True)
        return self.venv_python

    # ------------------------------------------------------------------ sandbox
    def write_bin(self, bin_dir: Path, workspace: Path, skills=()) -> None:
        """Helper commands available to agents: `hist` (history) and `ci-run` (local test runs)."""
        bin_dir.mkdir(parents=True, exist_ok=True)
        py = _posix(sys.executable)
        (bin_dir / "hist").write_text(
            f'#!/usr/bin/env bash\nPYTHONPATH="{_posix(PROJECT_ROOT)}" exec "{py}" -m ospm.replay.hist "$@"\n',
            encoding="utf-8", newline="\n")
        venv_py = _posix(self.venv_python) if self.venv_python.exists() else ""
        ci = ("#!/usr/bin/env bash\n"
              "# Run the project's test-suite locally on this workspace, e.g.: ci-run testing/test_foo.py -k bar\n"
              + (f'cd "{_posix(workspace)}" && PYTHONPATH="{_posix(workspace)}/src" exec timeout 600 "{venv_py}" '
                 '-m pytest -p no:cacheprovider -q --no-header -x "$@"\n' if venv_py else
                 'echo "local CI is not set up (run: ospm replay setup-ci)"; exit 2\n'))
        (bin_dir / "ci-run").write_text(ci, encoding="utf-8", newline="\n")

    def environment(self, workspace: Path, as_of: str, *, pr: int | None = None, pr_sha: str = "",
                    sha: str = "", base: str = "", timeout: int = 180, skills=()) -> LocalEnvironment:
        bin_dir = workspace.parent / f"{workspace.name}-bin"
        self.write_bin(bin_dir, workspace, skills)
        env = {"OSPM_HIST_DB": str(self.db_path.resolve()), "OSPM_HIST_ASOF": as_of,
               "OSPM_HIST_GIT": str(self.git_dir.resolve()), "OSPM_HIST_SHA": sha}
        if pr is not None:
            env.update({"OSPM_HIST_PR": str(pr), "OSPM_HIST_PR_SHA": pr_sha, "OSPM_HIST_PR_BASE": base})
        return LocalEnvironment(cwd=str(workspace), env=env, bin_dirs=[str(bin_dir)], timeout=timeout)


def disk_usage(path: str | Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, f))
    return total
