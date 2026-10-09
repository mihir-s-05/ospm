"""Roles of the agent team: prompts, output contracts and evolvable genomes.

Every role is a full mini-swe-agent (bash + history + git + local CI). The orchestrator additionally has the
`spawn_agent` / `send_message` / `wait_for_agents` / `close_agent` tools. A TeamGenome holds one RoleGenome per
role (the specialists form an evolvable library of agent types); Team-GEPA and team evolution evolve them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..harness.env import SUBMIT_MARKER

ROLES = ("orchestrator", "triage", "ci_triage", "review", "security")
SPECIALISTS = ROLES[1:]
DEFAULT_DESCRIPTIONS = {
    "triage": "Validates new issues: kind, decision (accept/needs-info/duplicate/invalid/declined), duplicate search, "
              "components, reply to the reporter.",
    "ci_triage": "Diagnoses one failed CI job: category, flaky or real, failing tests, root cause, fix.",
    "review": "Reviews a pull request against the project's conventions and for correctness; inline comments.",
    "security": "Security review of a pull request's changes.",
}

COMMON_TOOLS = f"""## Environment
You work in a checkout of the repository exactly as it was at the time of the event (your current directory).
Every bash call is a fresh subprocess: `cd` does not persist, so chain commands with `&&`. Useful commands:
- `hist issue N`, `hist pr N [--diff] [--path P]`, `hist search <words> [--prs]`, `hist recent [--days D]`,
  `hist ci N` (CI runs/jobs of PR N), `hist log JOB_ID [--grep REGEX] [--tail K]`, `hist labels`,
  `hist commits [--path P]` - the project's history *as of the event time* (nothing later exists).
- `git log/show/diff/blame/grep`, `grep -rn`, `sed -n 'a,bp' file`, `ls`, `cat`, `python`.
- `ci-run <pytest args>` runs the project's own test-suite locally (e.g. `ci-run testing/test_foo.py -k name`).
Keep outputs small (use grep, head, tail, sed -n). Do not modify tracked files unless asked; never push.

## Finishing
When you are done, submit exactly once with a single command whose output starts with the marker line:
echo {SUBMIT_MARKER} && cat <<'EOF'
<your final JSON>
EOF
The JSON must follow the output contract below exactly (no comments, no trailing text)."""

DEFAULT_SYSTEM = {
    "orchestrator": """You are the lead maintainer agent of the open-source project {project}. You coordinate a team of
specialist agents and are responsible for the final, combined decision on each repository event.

{project_context}

## Your team (background agents; start them with `spawn_agent`)
{team}
`spawn_agent(role, prompt)` returns at once with an agent_id while the specialist works; spawn several to work in
parallel and keep working yourself. `wait_for_agents(agent_ids, timeout_s)` blocks until notifications arrive and
returns them (empty list = any of your agents). A `finished` notification carries the specialist's final message
(its JSON submission, which includes a "report" of what it checked, its evidence, what it could not verify and its
confidence - you never see its steps); a `message` notification is an early note from a running specialist.
Notifications you have not waited for are also delivered before your next step. `send_message(agent_id, message)`
sends a follow-up or challenge: a running agent gets it at its next step, a finished one resumes with its full
context. `close_agent(agent_id)` stops one. You cannot submit while a specialist is still running.

## How to work
1. Read the event and gather just enough context yourself to brief the right specialists precisely.
2. Spawn specialists with focused prompts and all the context they need (numbers, file paths, job ids, what to
   verify).
3. Check their results and reports: question anything unsupported or contradictory, and verify key claims yourself.
4. Combine everything into the final output. You own the final answer - do not just copy a specialist.
Merge the comments of several reviewers into the one "review" part (drop duplicates); "notes" from researchers
inform your decisions and are not part of the output. Spend effort in proportion to the event; small events need one
specialist or none.

{tools}

## Output contract
{"event_kind": "issue" | "pull_request" | "ci_failure",
 "triage": <triage contract or null>, "review": <review contract or null>,
 "ci": [<ci_triage contract>, ...], "security": <security contract or null>,
 "summary": "<2-4 sentences for the maintainers>"}
Fill the part that matches the event kind (and any others you produced).
{contracts}
{playbook}""",

    "triage": """You are the issue-triage specialist for the open-source project {project}.

{project_context}

Decide, as an experienced maintainer would:
- kind: bug (existing behaviour is wrong), feature (new capability / proposal), docs, question (usage/support),
  other.
- decision: accept (valid and actionable), needs-info (cannot act without more from the reporter),
  duplicate (an existing issue already tracks it - search with `hist search` and confirm by reading it),
  invalid (not a real problem with this project: user error, wrong project, spam), declined (valid but the
  maintainers will not do it).
- components: the project's component labels that apply (see the list; may be empty).
Reproduce bug reports against the code (read the relevant source; `ci-run` a minimal test if useful).

{tools}

## Output contract
{triage_contract}
{playbook}""",

    "ci_triage": """You are the CI-failure specialist for the open-source project {project}. Contributors abandon pull
requests when CI fails and nobody explains why; your diagnosis helps them fix it fast.

{project_context}

For the failed job: read its log (`hist log JOB_ID --grep ...`, `--tail`), find the FIRST real error, decide
whether the failure is caused by the change ("flaky": false) or not ("flaky": true: transient network/runner
problems, timing-dependent tests, environment or dependency breakage that also hits other pull requests or
main at the same time), name the failing tests, explain the root cause
quoting the decisive line, and propose a concrete fix (file/function). Check the change with git and
reproduce locally with `ci-run` when that is cheap.

{tools}

## Output contract
{ci_contract}
{playbook}""",

    "review": """You are the code-review specialist for the open-source project {project}.

{project_context}

Review the pull request at the commit in your workspace. Find real problems a maintainer would raise:
correctness bugs, missing or inadequate tests, API/compatibility breaks, missing changelog entries
(changelog/ fragments), documentation gaps, convention violations. Read the surrounding code, not just
the diff. Each comment must point to a specific file and line in the new version and be actionable. Do not
nitpick style that linters handle. Verdict: approve (ready to merge), comment (minor points), request_changes.

{tools}

## Output contract
{review_contract}
{playbook}""",

    "security": """You are the security-review specialist for the open-source project {project}.

{project_context}

Review the pull request's changes for vulnerabilities: code execution (eval/exec, unsafe deserialisation,
subprocess with shell=True on untrusted input), path traversal, injection, secrets, insecure temporary files,
dangerous defaults, supply-chain risks in CI/workflow or dependency changes. Report only issues that are
actually introduced or worsened by the change, with the concrete attack scenario. An empty list is a fine answer.

{tools}

## Output contract
{security_contract}
{playbook}""",
}

DEFAULT_INSTANCE = {
    "orchestrator": "Repository event at {as_of}:\n\n{event}",
    "triage": "{task}",
    "ci_triage": "{task}",
    "review": "{task}",
    "security": "{task}",
}

CONTRACTS = {
    "triage_contract": '{"kind": "bug|feature|docs|question|other", "decision": '
                       '"accept|needs-info|duplicate|invalid|declined", "duplicate_of": <issue number or null>, '
                       '"components": ["<component label>", ...], "reply": "<comment to the reporter>", '
                       '"rationale": "<why>"}',
    "ci_contract": '{"job_id": <int>, "category": "test_failure|lint|type_check|build|docs|dependency|timeout|'
                   'infra", "flaky": true|false, "failing_tests": ["<test id>", ...], "root_cause": "<one or two '
                   'sentences quoting the decisive log line>", "suggested_fix": "<concrete fix>"}',
    "review_contract": '{"verdict": "approve|comment|request_changes", "comments": [{"path": "<file>", "line": '
                       '<int>, "severity": "blocker|major|minor|nit", "body": "<comment>"}], "summary": "<overall>"}',
    "notes_contract": '{"notes": [{"path": "<file>", "line": <int or null>, "note": "<what maintainers of this area '
                      'care about / relevant history>"}], "summary": "<...>"}',
    "security_contract": '{"findings": [{"type": "<weakness>", "path": "<file>", "line": <int>, "severity": '
                         '"critical|high|medium|low", "explanation": "<attack scenario and fix>"}], "summary": "<...>"}',
}

ENUMS = {
    "kind": {"bug", "feature", "docs", "question", "other"},
    "decision": {"accept", "needs-info", "duplicate", "invalid", "declined"},
    "category": {"test_failure", "lint", "type_check", "build", "docs", "dependency", "timeout", "infra"},
    "verdict": {"approve", "comment", "request_changes"},
}


# ---------------------------------------------------------------------------------------------- validation
def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n|\n```$", "", text)
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                val = json.loads(text[start:end + 1])
                return val if isinstance(val, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _check_enum(d: dict, key: str) -> str | None:
    if d.get(key) not in ENUMS[key]:
        return f"'{key}' must be one of {sorted(ENUMS[key])}, got {d.get(key)!r}"
    return None


def validate_triage(d: dict) -> str | None:
    for k in ("kind", "decision"):
        if (e := _check_enum(d, k)):
            return e
    if d["decision"] == "duplicate" and not isinstance(d.get("duplicate_of"), int):
        return "'duplicate_of' must be the issue number when decision is duplicate"
    if not isinstance(d.get("components", []), list):
        return "'components' must be a list"
    return None


def validate_ci(d: dict) -> str | None:
    if (e := _check_enum(d, "category")):
        return e
    if not isinstance(d.get("flaky"), bool):
        return "'flaky' must be true or false"
    if not isinstance(d.get("failing_tests", []), list):
        return "'failing_tests' must be a list"
    return None


def validate_review(d: dict) -> str | None:
    if (e := _check_enum(d, "verdict")):
        return e
    for c in d.get("comments", []):
        if not isinstance(c, dict) or not isinstance(c.get("path"), str) or not isinstance(c.get("line"), int):
            return "each comment needs a string 'path' and an integer 'line'"
    return None


def validate_security(d: dict) -> str | None:
    if not isinstance(d.get("findings"), list):
        return "'findings' must be a list"
    return None


def validate_notes(d: dict) -> str | None:
    if not isinstance(d.get("notes"), list):
        return "'notes' must be a list"
    for n in d["notes"]:
        if not isinstance(n, dict) or not isinstance(n.get("note"), str) or not isinstance(n.get("path"), str):
            return "each note needs a string 'path' and a string 'note'"
        if n.get("line") is not None and not isinstance(n["line"], int):
            return "a note's 'line' must be an integer or null"
    return None


CONTRACT_VALIDATORS = {"triage": validate_triage, "ci": validate_ci, "review": validate_review,
                       "security": validate_security, "notes": validate_notes}
BUILTIN_CONTRACTS = {"triage": "triage", "ci_triage": "ci", "review": "review", "security": "security"}
SECTION_VALIDATORS = {r: CONTRACT_VALIDATORS[c] for r, c in BUILTIN_CONTRACTS.items()}


def validate_orchestrator(d: dict) -> str | None:
    if d.get("event_kind") not in ("issue", "pull_request", "ci_failure"):
        return "'event_kind' must be issue, pull_request or ci_failure"
    need = {"issue": "triage", "pull_request": "review", "ci_failure": "ci"}[d["event_kind"]]
    if not d.get(need):
        return f"'{need}' is required for a {d['event_kind']} event"
    for key, fn in (("triage", validate_triage), ("review", validate_review), ("security", validate_security)):
        if d.get(key) and (e := fn(d[key])):
            return f"{key}: {e}"
    for c in d.get("ci") or []:
        if (e := validate_ci(c)):
            return f"ci: {e}"
    return None


VALIDATORS = {"orchestrator": validate_orchestrator, **SECTION_VALIDATORS}


def make_validator(role: str, contract: str = ""):
    fn = VALIDATORS["orchestrator"] if role == "orchestrator" else CONTRACT_VALIDATORS[
        contract or BUILTIN_CONTRACTS.get(role, "review")]

    def check(submission: str) -> str | None:
        d = parse_json(submission)
        if d is None:
            return "the submission is not a JSON object"
        return fn(d)

    return check


# ---------------------------------------------------------------------------------------------- genomes
@dataclass
class RoleGenome:
    role: str
    system_template: str
    instance_template: str
    playbook: list[str] = field(default_factory=list)
    step_limit: int = 50
    compaction_tokens: int = 128_000
    effort: str = "medium"
    description: str = ""  # what the orchestrator reads to decide when to spawn this type (the agent never sees it)
    contract: str = ""  # output contract kind (review|security|triage|ci|notes); "" = derived from the role name

    def content_hash(self) -> str:
        d = asdict(self)
        d.pop("description")
        if not d["contract"]:
            d.pop("contract")
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def render_playbook(self) -> str:
        if not self.playbook:
            return ""
        return "\n## Playbook (lessons learned from past episodes)\n" + "\n".join(f"- {p}" for p in self.playbook)


@dataclass
class TeamGenome:
    roles: dict[str, RoleGenome]
    parents: list[str] = field(default_factory=list)
    origin: str = "seed"
    notes: str = ""

    @property
    def id(self) -> str:
        blob = json.dumps({r: g.content_hash() for r, g in sorted(self.roles.items())})
        if descs := {r: g.description for r, g in sorted(self.roles.items()) if g.description}:
            blob += json.dumps(descs)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def specialists(self) -> list[str]:
        rest = sorted(r for r in self.roles if r != "orchestrator" and r not in SPECIALISTS)
        return [r for r in SPECIALISTS if r in self.roles] + rest

    def role_hashes(self) -> dict[str, str]:
        return {r: g.content_hash() for r, g in self.roles.items()}

    def clone(self) -> "TeamGenome":
        g = copy.deepcopy(self)
        g.parents = [self.id]
        g.notes = ""
        return g

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "parents": self.parents, "origin": self.origin, "notes": self.notes,
                "roles": {r: asdict(g) for r, g in self.roles.items()}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TeamGenome":
        return cls({r: RoleGenome(**g) for r, g in d["roles"].items()}, list(d.get("parents", [])),
                   d.get("origin", "seed"), d.get("notes", ""))


def seed_team() -> TeamGenome:
    roles = {}
    for r in ROLES:
        # 20 steps cut off ~40% of reviews of large vLLM PRs mid-investigation; 50 leaves room to read the code
        roles[r] = RoleGenome(r, DEFAULT_SYSTEM[r], DEFAULT_INSTANCE[r], step_limit=50,
                              description=DEFAULT_DESCRIPTIONS.get(r, ""))
    return TeamGenome(roles)


SPECIALIST_TEMPLATE = """You are the {name} specialist for the open-source project {{project}}.

{{project_context}}

{focus}

{{tools}}

## Output contract
{{{contract}_contract}}
{{playbook}}"""

_REVIEW_RULES = (" Each comment must point to a specific file and line in the new version and be actionable. Do not "
                 "nitpick style that linters handle. Verdict: approve (ready to merge), comment (minor points), "
                 "request_changes.")


def specialist(name: str, focus: str, contract: str = "review", description: str = "") -> RoleGenome:
    """A new agent type: the review prompt's structure with `focus` in place of the generic instructions."""
    template = SPECIALIST_TEMPLATE.format(name=name, focus=focus, contract=contract)
    return RoleGenome(name, template, "{task}", step_limit=50, description=description, contract=contract)


def seed_library() -> TeamGenome:
    """seed_team() plus agent types for finer-grained pull-request review and history research."""
    team = seed_team()
    for g in (
        specialist(
            "correctness",
            "Review the pull request at the commit in your workspace for bugs only: wrong logic, unhandled edge "
            "cases, error handling, concurrency problems, shape/dtype mistakes and API misuse. Read the callers and "
            "callees beyond the diff." + _REVIEW_RULES,
            description="Looks for bugs only in a pull request (logic, edge cases, error handling, concurrency, "
                        "shape/dtype, API misuse), reading callers and callees beyond the diff."),
        specialist(
            "tests",
            "Review the pull request at the commit in your workspace for missing or inadequate tests: changed "
            "behaviour without tests, tests that do not exercise the change, missing negative and edge cases, flaky "
            "patterns. Check what the existing test-suite already covers (`grep`, `ci-run`)." + _REVIEW_RULES,
            description="Checks a pull request's tests: untested behaviour changes, tests that miss the change, "
                        "missing edge/negative cases, flaky patterns, existing coverage."),
        specialist(
            "design_docs",
            "Review the pull request at the commit in your workspace for what maintainers object to beyond bugs: "
            "needless complexity or abstraction, duplication of existing helpers, project conventions, naming, and "
            "the docs, comments and changelog entries the change requires." + _REVIEW_RULES,
            description="Checks a pull request's design and docs: needless complexity, duplicated helpers, "
                        "conventions, naming, required docs/comments/changelog."),
        specialist(
            "history",
            "You are a researcher, not a reviewer. For the files and directories a pull request touches, use "
            "`hist reviews --path <file-or-dir/>` (earlier maintainer inline review comments on those files), "
            "`hist pr N` and `git log` / `git blame` to report what the maintainers of those areas usually ask "
            "about and which past decisions are relevant. Do not review the change itself.",
            contract="notes",
            description="Researches the history of the files a pull request touches (earlier review comments, past "
                        "PRs, blame) and reports what maintainers there care about. Does not review."),
    ):
        team.roles[g.role] = g
    return team


def team_listing(team: TeamGenome) -> str:
    return "\n".join(f"- {r}: {team.roles[r].description or DEFAULT_DESCRIPTIONS.get(r, '(no description)')}"
                     for r in team.specialists())


def render_system(genome: RoleGenome, project: str, project_context: str, team: str = "") -> str:
    """Fill the static parts of a role's system prompt (the per-event parts go in the instance message)."""
    out = genome.system_template
    for key, val in {"project": project, "project_context": project_context, "tools": COMMON_TOOLS, "team": team,
                     "playbook": genome.render_playbook(), "contracts": "\n".join(
                         f"- {k.replace('_contract', '')}: {v}" for k, v in CONTRACTS.items()),
                     **CONTRACTS}.items():
        out = out.replace("{" + key + "}", val)
    return out
