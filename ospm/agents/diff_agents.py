"""Pull-request diff analysis: convention review and security review (multi-label findings)."""

from __future__ import annotations

from typing import Any

from ..genome import Rule
from ..models import PullRequest, parse_diff
from ..text import normalize, truncate
from .base import Agent, FieldSpec


class DiffAgent(Agent):
    """Shared machinery: rules match the *added* lines; findings are located back to file/line."""

    finding_field = "findings"
    severity: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    default_params = {**Agent.default_params, "knn_weight": 0.3, "knn_k": 3, "min_sim": 0.25}

    def item_id(self, item: PullRequest) -> str:
        return item.id

    def item_text(self, item: PullRequest) -> str:
        chunks = []
        for f in parse_diff(item.diff):
            if not f.added:
                continue
            chunks.append(f"# file: {f.path}")
            chunks.extend(line for _, line in f.added)
        return "\n".join(chunks)

    def render_item(self, item: PullRequest) -> str:
        return truncate(f"PR #{item.number}: {item.title}\n{item.body}\n\nDiff:\n{item.diff}", 12000)

    def locate(self, item: PullRequest, label: str, fired: list[Rule]) -> list[dict[str, Any]]:
        """Find the added lines that triggered rules for `label`."""
        hits: list[dict[str, Any]] = []
        rules = [r for r in fired if r.value == label]
        for f in parse_diff(item.diff):
            for lineno, line in f.added:
                lower = line.lower()
                norm = normalize(line)
                if any(r.matches(lower, norm) for r in rules):
                    hits.append({"path": f.path, "line": lineno, "snippet": line.strip()[:160]})
        return hits[:5]

    def extras(self, item: PullRequest, text: str, labels: dict[str, Any], fired) -> dict[str, Any]:
        comments = []
        for label in labels[self.finding_field]:
            locs = self.locate(item, label, fired) or [{"path": "", "line": 0, "snippet": ""}]
            for loc in locs:
                comments.append({
                    "finding": label,
                    "severity": self.severity.get(label, "minor"),
                    "path": loc["path"],
                    "line": loc["line"],
                    "message": self.descriptions.get(label, label),
                })
        return {"comments": comments}

    def extra_schema(self) -> dict[str, dict]:
        labels = list(self.field(self.finding_field).labels)
        return {
            "comments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "finding": {"type": "string", "enum": labels + ["other"]},
                        "severity": {"type": "string", "enum": ["blocker", "major", "minor", "nit"]},
                        "path": {"type": "string"},
                        "line": {"type": "integer"},
                        "message": {"type": "string"},
                    },
                    "required": ["finding", "severity", "path", "line", "message"],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string"},
        }

    def merge_llm_extras(self, item, text, labels, fired, heuristic_extras, llm_extras):
        merged = dict(self.extras(item, text, labels, fired))
        if llm_extras.get("comments"):
            comments = llm_extras["comments"]
            # drop generic "other" comments that restate a labelled finding on the same line
            labelled = {(c["path"], c["line"]) for c in comments if c.get("finding") != "other"}
            merged["comments"] = [c for c in comments
                                  if c.get("finding") != "other" or (c.get("path"), c.get("line")) not in labelled]
        if llm_extras.get("summary"):
            merged["summary"] = llm_extras["summary"]
        return merged


class ReviewAgent(DiffAgent):
    name = "review"
    title = "code reviewer"
    fields = (
        FieldSpec("findings", ("debug_print", "bare_except", "todo_without_issue", "missing_tests",
                               "breakpoint_left", "mutable_default_arg", "wildcard_import", "missing_type_hints"),
                  multi=True,
                  description="Convention violations introduced by the ADDED lines (ignore removed lines). "
                              "debug_print = print() in library code (use logging); bare_except = `except:` "
                              "without an exception type; todo_without_issue = TODO/FIXME without an issue "
                              "reference like #123; missing_tests = library code changed but no test file changed; "
                              "breakpoint_left = breakpoint()/pdb left in; mutable_default_arg = list/dict/set "
                              "default argument; wildcard_import = `from x import *`; missing_type_hints = new "
                              "public function without parameter/return annotations."),
    )
    severity = {"debug_print": "minor", "bare_except": "major", "todo_without_issue": "nit",
                "missing_tests": "major", "breakpoint_left": "blocker", "mutable_default_arg": "major",
                "wildcard_import": "minor", "missing_type_hints": "minor"}
    descriptions = {
        "debug_print": "Use the module logger instead of print() in library code.",
        "bare_except": "Catch a specific exception type; a bare `except:` also swallows KeyboardInterrupt/SystemExit.",
        "todo_without_issue": "Reference a tracking issue in TODO/FIXME comments (e.g. `TODO(#123): ...`).",
        "missing_tests": "Library code changed without accompanying tests; please add or update tests.",
        "breakpoint_left": "Remove the debugger breakpoint before merging.",
        "mutable_default_arg": "Mutable default arguments are shared between calls; default to None instead.",
        "wildcard_import": "Avoid wildcard imports; import the names you use explicitly.",
        "missing_type_hints": "Public functions should have type annotations for parameters and return value.",
    }
    seed_instructions = """
Review the change against the project's coding conventions. Report only problems introduced by added lines,
and only those you are confident about; precision matters because false alarms waste contributors' time.
For each problem cite the file and line and explain the fix briefly. Use the conventions listed in the
project context; also note business-logic concerns in the summary.
"""
    seed_rules = (
        {"id": "print-call", "field": "findings", "value": "debug_print", "pattern": r"^\s*print\(", "weight": 1.0},
        {"id": "bare-except", "field": "findings", "value": "bare_except", "pattern": r"except\s*:", "weight": 1.0},
        {"id": "breakpoint", "field": "findings", "value": "breakpoint_left",
         "any_of": ["breakpoint", "pdb set_trace"], "weight": 1.0},
        {"id": "wildcard", "field": "findings", "value": "wildcard_import",
         "pattern": r"^\s*from\s+\S+\s+import\s+\*", "weight": 1.0},
    )

    def base_scores(self, item: PullRequest, text: str) -> dict[str, dict[str, float]]:
        files = parse_diff(item.diff)
        lib_changed = sum(len(f.added) for f in files if self.project.is_library_path(f.path) and f.path.endswith(".py"))
        tests_changed = any(self.project.is_test_path(f.path) for f in files)
        out: dict[str, float] = {}
        if lib_changed >= 3 and not tests_changed:
            out["missing_tests"] = 1.0
        return {"findings": out}


class SecurityAgent(DiffAgent):
    name = "security"
    title = "application security reviewer"
    fields = (
        FieldSpec("findings", ("hardcoded_secret", "sql_injection", "command_injection", "unsafe_deserialization",
                               "insecure_tls", "weak_crypto", "path_traversal", "code_eval"),
                  multi=True,
                  description="Security weaknesses introduced by the ADDED lines. hardcoded_secret = credentials/"
                              "keys/tokens in source; sql_injection = SQL built from untrusted input by string "
                              "formatting; command_injection = shell commands built from untrusted input "
                              "(shell=True, os.system); unsafe_deserialization = pickle/marshal/yaml.load on "
                              "untrusted data; insecure_tls = disabled certificate verification; weak_crypto = "
                              "MD5/SHA1 for security or `random` for secrets/tokens; path_traversal = file paths "
                              "built from untrusted input without normalisation/containment checks; code_eval = "
                              "eval/exec on untrusted input."),
    )
    severity = {k: "blocker" for k in ("hardcoded_secret", "sql_injection", "command_injection",
                                         "unsafe_deserialization", "code_eval")}
    severity.update({"insecure_tls": "major", "weak_crypto": "major", "path_traversal": "major"})
    descriptions = {
        "hardcoded_secret": "Credential committed to source; move it to configuration/secrets and rotate it.",
        "sql_injection": "SQL built by string formatting; use parameterised queries.",
        "command_injection": "Shell command built from untrusted input; pass an argument list without shell=True.",
        "unsafe_deserialization": "Deserialising untrusted data can execute code; use json or yaml.safe_load.",
        "insecure_tls": "TLS certificate verification is disabled.",
        "weak_crypto": "Weak hash/RNG for a security purpose; use hashlib.sha256/bcrypt or the secrets module.",
        "path_traversal": "Path built from untrusted input; resolve it and check it stays inside the base dir.",
        "code_eval": "eval/exec on untrusted input allows arbitrary code execution.",
    }
    seed_instructions = """
Act as a security reviewer for every pull request. Identify vulnerabilities introduced by the added lines,
considering where data comes from (user input, files, network) and where it flows. Report a finding only if
the vulnerable pattern is actually introduced; constant, trusted inputs are not injections. Explain the
exploit scenario and the safe alternative.
"""
    seed_rules = (
        {"id": "secret-assign", "field": "findings", "value": "hardcoded_secret",
         "pattern": r"(api[_-]?key|secret|password|passwd|token)\w*\s*=\s*['\"][^'\"]{8,}['\"]", "weight": 1.0},
        {"id": "verify-false", "field": "findings", "value": "insecure_tls", "any_of": ["verify false"], "weight": 1.0},
        {"id": "eval-call", "field": "findings", "value": "code_eval", "pattern": r"\b(eval|exec)\(", "weight": 1.0},
    )
