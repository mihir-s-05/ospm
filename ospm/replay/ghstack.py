"""ghstack pull requests (PyTorch): what a reviewer actually saw.

A ghstack PR targets a synthetic `gh/<user>/<n>/base` branch, not `main`, and lands on `main` through a bot. Neither
branch survives, but the commits do (they are reachable from `refs/pull/<n>/head`), and ghstack marks them:

- base commits:  first line "... (base update)", body "[ghstack-poisoned]"
- head commits:  "[ghstack-poisoned]" without that marker; an update whose base moved is a merge whose second
                 parent is the new base commit, an update on an unchanged base has a single parent (the previous head)

So the base at review time is the second parent of the nearest merge on the reviewed head's first-parent chain, or,
if there is none, the parent of the chain's first head commit.
"""

from __future__ import annotations

from typing import Callable

BASE_MARK = "(base update)"
POISON = "[ghstack-poisoned]"


def is_ghstack(base_ref: str | None) -> bool:
    return bool(base_ref) and base_ref.startswith("gh/") and base_ref.endswith("/base")


def target_branch(base_ref: str | None) -> str:
    """The branch a PR lands on: ghstack PRs land on main."""
    return "main" if is_ghstack(base_ref) else (base_ref or "main")


def _is_head(message: str) -> bool:
    first = message.strip().splitlines()[0] if message.strip() else ""
    return POISON in message and BASE_MARK not in first


def base_of(git: Callable[..., str], sha: str, limit: int = 500) -> str:
    """Base commit the ghstack head `sha` was reviewed against, or "" when `sha` is not a ghstack head commit.

    `git(*args)` runs git in the mirror and returns stdout.
    """
    cur = sha
    for _ in range(limit):
        parents, _, message = git("log", "-1", "--format=%P%x00%B", cur).partition("\x00")
        if not _is_head(message):
            return "" if cur == sha else cur  # walked past the first head commit: this is the original base
        parents = parents.split()
        if len(parents) >= 2:
            return parents[1]
        if not parents:
            return ""
        cur = parents[0]
    return ""
