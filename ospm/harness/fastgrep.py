"""Recursive grep through `git grep`.

On Windows, `grep -R` walks every file through the MSYS layer: a whole-tree search of a 30k-file checkout (Home
Assistant) takes 100-400 s and runs past the command timeout, while `git grep` answers in a second or two. The
sandbox's `grep` wrapper sends recursive searches here; whatever cannot be translated faithfully exits with
`FALLBACK` and runs the real grep. Only tracked files are searched (`--untracked` triples the time), which in a
replay workspace is every file but the ones an agent writes itself.

    python -m ospm.harness.fastgrep -rn "async_setup_entry" homeassistant --include=*.py
"""

from __future__ import annotations

import os
import subprocess
import sys

FALLBACK = 254
_FLAGS = {"n": "-n", "i": "-i", "l": "-l", "L": "-L", "c": "-c", "h": "-h", "H": "-H", "o": "-o", "w": "-w",
          "v": "-v", "E": "-E", "F": "-F", "G": "-G", "P": "-P", "I": "-I", "a": "--text", "q": "-q", "x": "-x"}
_DROP = set("rRs")  # recursion is implied; git grep has no -s (it does not complain about unreadable files)
_VALUED = {"e": "-e", "A": "-A", "B": "-B", "C": "-C", "m": "-m", "f": "-f"}
_LONG = {"--line-number": "-n", "--ignore-case": "-i", "--files-with-matches": "-l", "--files-without-match": "-L",
         "--count": "-c", "--no-filename": "-h", "--with-filename": "-H", "--only-matching": "-o",
         "--word-regexp": "-w", "--invert-match": "-v", "--extended-regexp": "-E", "--fixed-strings": "-F",
         "--basic-regexp": "-G", "--perl-regexp": "-P", "--text": "--text", "--quiet": "-q", "--silent": "-q",
         "--line-regexp": "-x", "--recursive": "", "--dereference-recursive": "", "--no-messages": "",
         "--color": "", "--colour": "", "--color=never": "", "--colour=never": "", "--color=auto": "",
         "--colour=auto": "", "--binary-files=without-match": "-I"}
_LONG_VALUED = {"--regexp": "-e", "--after-context": "-A", "--before-context": "-B", "--context": "-C",
                "--max-count": "-m", "--file": "-f", "--include": "include", "--exclude": "exclude",
                "--exclude-dir": "exclude-dir"}


def translate(argv: list[str], cwd: str = ".", missing: list[str] | None = None) -> list[str] | None:
    """`git grep` arguments equivalent to a recursive `grep argv` run in `cwd`, or None to use the real grep.
    Operands that do not exist are left out and appended to `missing` (grep reports them and searches the rest)."""
    opts, operands, pattern_given, recursive = [], [], False, False
    include, exclude = [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            operands += argv[i + 1:]
            break
        if a.startswith("--"):
            name, eq, val = a.partition("=")
            if a in _LONG:
                opts += [_LONG[a]] if _LONG[a] else []
                recursive |= a in ("--recursive", "--dereference-recursive")
            elif name in _LONG_VALUED:
                if not eq:
                    i += 1
                    if i >= len(argv):
                        return None
                    val = argv[i]
                kind = _LONG_VALUED[name]
                if kind == "include":
                    include.append(val)
                elif kind in ("exclude", "exclude-dir"):
                    exclude.append(f"**/{val}" if kind == "exclude" else f"**/{val}/**")
                else:
                    opts += [kind, val]
                    pattern_given |= kind in ("-e", "-f")
            else:
                return None
        elif a.startswith("-") and len(a) > 1:
            j = 1
            while j < len(a):
                ch = a[j]
                if ch in _VALUED:
                    val = a[j + 1:]
                    if not val:
                        i += 1
                        if i >= len(argv):
                            return None
                        val = argv[i]
                    opts += [_VALUED[ch], val]
                    pattern_given |= ch in "ef"
                    break
                if ch.isdigit():  # -5 means -C 5
                    opts += ["-C", a[j:]]
                    break
                if ch in _FLAGS:
                    opts.append(_FLAGS[ch])
                elif ch in _DROP:
                    recursive |= ch in "rR"
                else:
                    return None
                j += 1
        else:
            operands.append(a)
        i += 1
    if not recursive:
        return None  # a plain grep reads stdin or the named files: nothing to speed up
    if not pattern_given:
        if not operands:
            return None
        opts += ["-e", operands.pop(0)]
    paths, absent = [], []
    for p in operands or ["."]:
        if p.startswith(("/", "\\", "~")) or os.path.isabs(p):
            return None  # outside the checkout
        if not os.path.exists(os.path.join(cwd, p)):
            absent.append(p)
            continue
        rel = os.path.relpath(os.path.join(cwd, p), cwd).replace("\\", "/")
        if rel.startswith(".."):
            return None
        paths.append("" if rel == "." else rel)
    if not paths:
        return None  # nothing to search: grep reports the missing paths itself
    if missing is not None:
        missing += absent
    spec = []
    for p in paths:
        if include and not os.path.isfile(os.path.join(cwd, p or ".")):
            spec += [f":(glob){p + '/' if p else ''}**/{g}" for g in include]
        else:
            spec.append(p or ".")
    spec += [f":(exclude,glob){g}" for g in exclude]
    return ["git", "grep", "--no-color", *opts, "--", *spec]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    missing: list[str] = []
    cmd = translate(argv, missing=missing)
    inside = cmd and subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True,
                                    text=True).stdout.strip() == "true"
    if not inside:
        return FALLBACK
    for p in missing:
        print(f"grep: {p}: No such file or directory", file=sys.stderr, flush=True)
    rc = subprocess.run(cmd).returncode
    return 2 if missing and not ("-q" in cmd and rc == 0) else rc  # as grep: 2 after an error, unless -q matched


if __name__ == "__main__":
    raise SystemExit(main())
