"""CI failure triage: categorise a failed job, extract the root cause and suggest a fix."""

from __future__ import annotations

import re
from typing import Any

from ..models import CIRun
from ..text import truncate
from .base import Agent, FieldSpec

_TS_PREFIX = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+)")
_SALIENT = re.compile(
    r"(error|failed|failure|exception|traceback|fatal|panic|assert|timed? ?out|cancel|exit code [1-9]|"
    r"##\[error\]|could not|cannot|not found|unable to|denied|refused|reset by peer|vulnerab|cve-|e\d{3}\b|"
    r"\bF\d{3}\b|would reformat|incompatible)",
    re.IGNORECASE,
)
_FAILED_TEST_PATTERNS = (
    re.compile(r"^FAILED\s+(\S+::\S+)"),  # pytest summary
    re.compile(r"^\s*--- FAIL: (\S+)"),  # go test
    re.compile(r"^\s*● (.+?) › (.+)$"),  # jest
    re.compile(r"^\s*(\S+Test\.\w+)\s+FAILED"),  # gradle/junit
)

FIX_TEMPLATES = {
    "test_failure": "Run the failing test(s) locally (`{repro}`), compare the assertion's expected and actual values, "
                    "and fix the code path or update the test if the behaviour change is intended.",
    "lint": "Run the project's linters/formatters locally (e.g. `ruff check --fix . && ruff format .`) and commit the result.",
    "type_check": "Run the type checker locally (e.g. `mypy src`) and fix the reported annotation/type mismatches.",
    "build": "Reproduce the build locally from a clean checkout; check for syntax errors, missing files in the package "
             "manifest, or compiler errors shown above.",
    "dependency": "Check the dependency spec: pin or relax the conflicting version, regenerate the lock file, and make "
                  "sure the package name/version exists on the index.",
    "timeout": "Find the slow or hanging step (a deadlock, network wait, or an unbounded loop); add timeouts to tests or "
               "split the job.",
    "infra_flaky": "This looks like a transient infrastructure problem (network/runner). Re-run the job; if it "
                   "recurs, add retries or caching around the flaky step.",
    "security_scan": "Upgrade the vulnerable dependency to a patched version (see the advisory above), or remove the "
                     "flagged pattern; re-run the security scan.",
}


def strip_ts(line: str) -> str:
    return _TS_PREFIX.sub("", line).rstrip()


def salient_lines(log: str, max_lines: int = 40, tail: int = 12) -> list[str]:
    """Error-bearing lines (with a little traceback context) plus the tail of the log."""
    lines = [strip_ts(l) for l in log.splitlines()]
    keep: list[int] = []
    in_tb = 0
    for i, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            in_tb = 12
        if in_tb > 0 or _SALIENT.search(line):
            keep.append(i)
        in_tb = max(0, in_tb - 1)
    keep.extend(range(max(0, len(lines) - tail), len(lines)))
    seen: set[int] = set()
    out: list[str] = []
    for i in sorted(keep):
        if i not in seen and lines[i].strip():
            seen.add(i)
            out.append(lines[i])
    if len(out) > max_lines:
        out = out[: max_lines - tail] + ["..."] + out[-tail:]
    return out


def failing_tests(log: str) -> list[str]:
    found: list[str] = []
    for raw in log.splitlines():
        line = strip_ts(raw)
        for pat in _FAILED_TEST_PATTERNS:
            m = pat.search(line)
            if m:
                name = " › ".join(g for g in m.groups() if g)
                if name not in found:
                    found.append(name)
    return found


_BOILERPLATE = re.compile(
    r"(process completed with exit code|a complete log of this run|see above for more details|"
    r"^error: process completed|exited with code|the operation was canceled|##\[error\]process completed|"
    r"^run |^> |^\$ )",
    re.IGNORECASE,
)
_SUCCESS = re.compile(r"(✓|\bPASSED\b|\bpassed\b|\.\.\. ok\b|^ok\s|Passed$)")
_SPECIFIC = re.compile(
    r"(\S+\.\w+[:(]\d+|\b[A-Z]{1,3}\d{3,4}\b|error\[E\d+\]|AssertionError|\bassert\b|^FAILED |--- FAIL|●|"
    r"panicked at|ResolutionImpossible|ERESOLVE|No matching distribution|does not contain package|"
    r"changed significantly|CVE-|GHSA-|leaks found|exceeded the maximum execution time|timed out|"
    r"^Failed: Timeout|ECONNRESET|lost communication|Service Unavailable|Bad Gateway|not found|unable to|"
    r"would reformat)",
)
_GENERIC = re.compile(r"(error|exception|failed|failure|fail:|fatal|panic|timeout|exceeded)", re.IGNORECASE)


def first_error(lines: list[str]) -> str:
    """The most informative early error line: specific diagnostics beat generic errors beat boilerplate."""
    best, best_score = "", -1
    for line in lines:
        text = line.strip()
        if not text or "Traceback (most recent call last)" in text:
            continue
        if _BOILERPLATE.search(text) or _SUCCESS.search(text):
            score = 0
        elif _SPECIFIC.search(text):
            score = 3
        elif _GENERIC.search(text):
            score = 2
        else:
            score = 1
        if score > best_score:
            best, best_score = text, score
    return best


class CITriageAgent(Agent):
    name = "ci_triage"
    title = "CI failure triage engineer"
    fields = (
        FieldSpec("category", ("test_failure", "lint", "type_check", "build", "dependency", "timeout",
                               "infra_flaky", "security_scan"), default="test_failure",
                  description="test_failure = a test assertion/exception; lint = style/format checker; type_check = "
                              "static type checker; build = compile/packaging/syntax failure; dependency = resolution "
                              "or install failure caused by the project's dependency spec; timeout = job or test "
                              "exceeded a time limit; infra_flaky = transient runner/network/service problem unrelated "
                              "to the change; security_scan = vulnerability/secret scanner finding."),
    )
    seed_instructions = """
Diagnose why a CI job failed so the contributor can fix it quickly instead of abandoning the pull request.
Find the FIRST real error (later errors are often consequences), name the root cause in one sentence that
quotes the decisive log line, and give a concrete, minimal fix. Distinguish failures caused by the change from
transient infrastructure problems, for which the right action is a re-run.
"""
    seed_rules = (
        {"id": "pytest-failed", "field": "category", "value": "test_failure",
         "any_of": ["assertionerror", "failed tests"], "weight": 1.0},
        {"id": "ruff-lint", "field": "category", "value": "lint",
         "any_of": ["ruff", "flake8", "would reformat"], "weight": 1.0},
        {"id": "mypy", "field": "category", "value": "type_check",
         "any_of": ["mypy", "incompatible type"], "weight": 1.0},
        {"id": "timeout", "field": "category", "value": "timeout",
         "any_of": ["timed out", "timeout"], "weight": 1.0},
        {"id": "pip-resolve", "field": "category", "value": "dependency",
         "any_of": ["resolutionimpossible", "no matching distribution"], "weight": 1.0},
    )

    def item_id(self, item: CIRun) -> str:
        return item.run_id or item.url or f"ci-{item.name}"

    def item_text(self, item: CIRun) -> str:
        return f"job: {item.name}\n" + "\n".join(salient_lines(item.log))

    def render_item(self, item: CIRun) -> str:
        return truncate(f"Job: {item.name} (conclusion: {item.conclusion})\n\nRelevant log lines:\n"
                        + "\n".join(salient_lines(item.log, max_lines=80, tail=25)), 9000)

    def base_scores(self, item: CIRun, text: str) -> dict[str, dict[str, float]]:
        cat: dict[str, float] = {"test_failure": 0.2}
        if failing_tests(item.log):
            cat["test_failure"] += 0.8
        if item.conclusion == "timed_out" or item.conclusion == "cancelled":
            cat["timeout"] = 1.5
        return {"category": cat}

    def extras(self, item: CIRun, text: str, labels: dict[str, Any], fired) -> dict[str, Any]:
        lines = salient_lines(item.log)
        tests = failing_tests(item.log)
        repro = "pytest " + " ".join(t.split(" › ")[0] for t in tests[:3]) if tests else "the failing command"
        cat = labels["category"]
        return {
            "root_cause": first_error(lines),
            "failing_tests": tests,
            "suggested_fix": FIX_TEMPLATES[cat].format(repro=repro),
            "retry_recommended": cat == "infra_flaky",
            "evidence": lines[:8],
        }

    def extra_schema(self) -> dict[str, dict]:
        return {
            "root_cause": {"type": "string"},
            "suggested_fix": {"type": "string"},
            "failing_tests": {"type": "array", "items": {"type": "string"}},
        }

    def merge_llm_extras(self, item, text, labels, fired, heuristic_extras, llm_extras):
        merged = super().merge_llm_extras(item, text, labels, fired, heuristic_extras, llm_extras)
        merged["retry_recommended"] = labels["category"] == "infra_flaky"
        return merged
