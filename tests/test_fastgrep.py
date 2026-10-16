"""Recursive grep through git grep (ospm.harness.fastgrep) and the sandbox's grep wrapper."""

from __future__ import annotations

import subprocess
import sys

import pytest

from ospm.harness.env import LocalEnvironment
from ospm.harness.fastgrep import translate
from ospm.replay.workspace import GREP_SCRIPT, PROJECT_ROOT, _posix


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def get_light():\n    return 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.json").write_text('{"get_light": 1}\n', encoding="utf-8")
    (tmp_path / "notes.txt").write_text("nothing here\n", encoding="utf-8")
    for cmd in (["init", "-q"], ["add", "."], ["-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "x"]):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_translation_keeps_flags_and_turns_include_exclude_into_pathspecs(repo):
    cwd = str(repo)
    assert translate(["-rn", "get_light", "."], cwd) == ["git", "grep", "--no-color", "-n", "-e", "get_light", "--",
                                                         "."]
    assert translate(["-R", "-i", "-A2", "x", "pkg", "--include=*.py"], cwd)[-2:] == ["--", ":(glob)pkg/**/*.py"]
    assert translate(["-rl", "--exclude-dir", "pkg", "-e", "-x"], cwd) == [
        "git", "grep", "--no-color", "-l", "-e", "-x", "--", ".", ":(exclude,glob)**/pkg/**"]


def test_missing_paths_are_left_out_and_reported(repo):
    missing = []
    assert translate(["-rn", "x", "pkg", "docs"], str(repo), missing)[-2:] == ["--", "pkg"] and missing == ["docs"]


@pytest.mark.parametrize("argv", [
    ["-n", "x", "notes.txt"],  # not recursive: plain grep
    ["-rz", "x", "."],  # -z has no git grep equivalent
    ["-r", "x", "/etc"],  # outside the checkout
    ["-r", "x", "missing"],  # nothing left to search: grep reports the error
    ["-r", "--null-data", "x"],
])
def test_untranslatable_searches_fall_back(repo, argv):
    assert translate(argv, str(repo)) is None


def test_the_sandbox_grep_wrapper_wins_over_the_system_grep_and_finds_matches(repo, tmp_path_factory):
    bin_dir = tmp_path_factory.mktemp("bin")
    (bin_dir / "grep").write_text(GREP_SCRIPT.format(root=_posix(PROJECT_ROOT), py=_posix(sys.executable)),
                                  encoding="utf-8", newline="\n")
    env = LocalEnvironment(cwd=str(repo), bin_dirs=[str(bin_dir)])
    assert env.execute("type grep").output.split()[-1].endswith(f"{bin_dir.name}/grep")
    assert env.execute("grep -rn get_light .").output.splitlines() == ["pkg/a.py:1:def get_light():",
                                                                      'pkg/b.json:1:{"get_light": 1}']
    assert env.execute("grep -rl get_light . --include=*.py").output.split() == ["pkg/a.py"]
    assert env.execute("printf 'a\\nb\\n' | grep -n b").output.strip() == "2:b"  # stdin: the real grep
    assert env.execute("grep -rn nosuch .").returncode == 1
    res = env.execute("grep -rn get_light pkg docs")
    assert res.returncode == 2 and "grep: docs: No such file or directory" in res.output and "pkg/a.py:1:" in res.output
    env.cleanup()
