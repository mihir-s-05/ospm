# The agent team, the replay environment, and Team-GEPA

This document covers the agentic half of `ospm`: tool-using agents built like mini-swe-agent v2, a primary
orchestrator that coordinates background specialists, a replay environment built from a real repository's history, and
**Team-GEPA**, a variant of GEPA for evolving multi-agent systems.

```
 replayed event (issue opened / PR ready for review / CI job failed), clock frozen at the event time
        │
        ▼
 ┌──────────────────────── orchestrator (mini-swe-agent + agent tools) ──────────────────────────────┐
 │  bash: hist issue/pr/ci/log/search · git log/show/blame/grep · ci-run (local pytest)               │
 │  spawn_agent(role, prompt) ──────┬─► triage     (mini-swe-agent) ─┐  background threads; the        │
 │  send_message / close_agent      ├─► review     (mini-swe-agent)  │  parent sees only each final    │
 │  wait_for_agents ◄── finished /  ├─► security   (mini-swe-agent)  ├─► message (JSON + "report"),    │
 │      message notifications       └─► ci_triage  (mini-swe-agent) ─┘  never the trace                │
 │  final JSON (triage / review / ci / security / summary) ──► scored against what really happened  │
 └────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │ hierarchical trace + replay-grounded feedback
        ▼
 Team-GEPA: blame the responsible role ──► rewrite that role ──► minibatch gate ──► cost-aware Pareto set
```

## 1. Harness (`ospm/harness/`), as in mini-swe-agent v2

| mini-swe-agent v2 | here |
|---|---|
| linear history; every step appends to the messages | `Agent._loop`; histories are append-only |
| native tool calling with a single `bash` tool (text-based ```` ```mswea_bash_command ```` mode as an alternative) | `BASH_TOOL`; `AgentConfig.text_mode=True` gives the text-based protocol |
| every action is an independent `subprocess.run` | `LocalEnvironment.execute`: a fresh `bash -c` per action |
| finish with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`; the remaining output is the submission | `extract_submission`; submissions are validated against the role's contract, and malformed ones are sent back to the model |
| `format_error_template`, step / cost / wall-time limits | `FORMAT_ERROR_*`, `AgentConfig.step_limit/cost_limit/wall_time_limit` |
| native provider messages (`extra` added) | `Conversation` keeps OpenAI Responses items / Anthropic content blocks as returned (reasoning and thinking round-trip unchanged) |

What the harness adds on top:

- **Context compaction.** When the context passes `compaction_tokens`, a summariser call condenses the
  transcript. The whole history is then replaced by one message holding the task, the summary and the last
  observation; nothing older is replayed. This "simple compaction" is the shape Anthropic recommends for
  client-side compaction with preserved thinking. It never happens mid tool round, and it is equally valid for
  OpenAI's stateless Responses input.
- **Extra tools.** These run concurrently when issued in the same turn. The orchestrator's agent tools (section 2)
  are such tools.
- **`Agent.send()`.** Continues a finished agent with a follow-up message, so the orchestrator can question or
  challenge a specialist that keeps its context.
- **`Agent.post()` / `Agent.close()`.** `post` queues a message for a running agent; the queue is drained
  (`conv.add_user`, plus a `message` step in the trace) before every step, and an empty queue changes nothing.
  `close` stops the agent at its next iteration with exit status `Closed`.
- **Hierarchical traces.** Sub-agent runs are nested under the agent that spawned them (resumed runs keep adding
  to the same child trace), and saved gzipped per episode (`ospm trace show FILE`).
- **Sandbox (best effort).** The child environment is rebuilt from an allow-list, so API keys, tokens, the
  user's HOME, gh credentials, SSH and git credential helpers are invisible. Commands have timeouts that kill
  the process tree, and a deny-list blocks pushes, `gh`, `sudo` and destructive commands. Network and
  filesystem isolation still need a container; swapping the environment class is all that takes, as in
  mini-swe-agent.

## 2. Team (`ospm/team/`)

There are five roles: `orchestrator`, `triage`, `ci_triage`, `review` and `security`. Each has a system prompt
(project context, tool documentation, an output contract, and a playbook slot), a step limit, a compaction
threshold and a reasoning effort. Together these make up a `RoleGenome`; a `TeamGenome` holds one per role.
`run_episode(genome, model, env_factory, entry_role=...)` runs either the full team or a single specialist (the
specialist-only baseline).

**Library of agent types.** The specialists are not a fixed list: a `TeamGenome` is a library of agent types, so an
evolution loop can add, remove and edit them. Each `RoleGenome` has a `description` (one or two sentences that only
the orchestrator reads; excluded from the role's content hash) and a `contract` (`review`, `security`, `triage`,
`ci` or `notes`; empty means derived from the built-in role name and is excluded from the hash, so cached episodes of
the built-in roles stay valid). The orchestrator's prompt holds a `{team}` placeholder filled by `team_listing(genome)`
with one `- name: description` line per specialist, and the `spawn_agent` role enum and each agent's output validator
come from the session's genome (`TeamGenome.specialists()`, `make_validator(role, contract)`). `TeamGenome.id` also
covers the descriptions. `specialist(name, focus, contract, description)` builds a new review-style type, and
`seed_library()` is `seed_team()` plus `correctness`, `tests`, `design_docs` (review contract) and `history` (a
researcher with the `notes` contract that mines `hist reviews --path` and `git blame`).

**Coordination.** The orchestrator manages specialists like Claude Code subagents; they run in background threads
(one `TeamSession` per event, a cap of `max_specialists` per parent):

| tool | effect |
|---|---|
| `spawn_agent(role, prompt)` | starts a specialist, returns `{"agent_id", "role", "status": "running"}` at once |
| `send_message(agent_id, message)` | a running child gets it at its next step (`delivered`); a finished child is resumed with its context (`resumed`); closed, unknown or foreign ids are errors |
| `wait_for_agents(agent_ids, timeout_s)` | blocks until a notification is pending, none of the agents runs any more, or the timeout (default 300 s, max 900 s); returns and consumes all pending notifications |
| `close_agent(agent_id)` | stops a child (`Closed`); it cannot be resumed |
| `message_parent(message)` | specialists spawned with a parent only; non-blocking note to the parent |

- **Notifications.** One queue per parent, holding `{"type": "finished", "agent_id", "role", "exit_status",
  "final_message"}` (the child's submission, parsed as JSON when possible) and `{"type": "message", "from", "role",
  "message"}`. They are consumed either by `wait_for_agents` or automatically as `<notification>{json}</notification>`
  user messages before the parent's next step, and never twice.
- **Work report.** The parent sees only a child's final message. A spawned specialist must therefore include a
  top-level `"report"` (what it checked, evidence per finding, what it could not verify, confidence); a validator
  rejects the submission without it and its task message carries a note saying so. Standalone specialists
  (`entry_role="review"` etc.) are unchanged: no note, no requirement.
- **Submission rules.** An agent cannot submit while one of its children is running (`wait_for_agents` or
  `close_agent` first). When the entry agent ends, `run_episode` closes the agents still running and joins their
  threads (30 s) before cleaning up the environments.

## 3. Replay environment (`ospm/replay/`)

- **`ospm mirror pytest-dev/pytest --since 2026-04-01 --out data/pytest`** builds:
  - a bare clone, plus the window's PR heads;
  - `history.db` with issues/PRs, label/close events, comments, reviews, inline review comments, CI runs,
    jobs for every failed run and re-run attempt, and failed-job logs (head and tail trimmed, zlib-compressed).

  It uses bulk repo-wide endpoints: about 2,000 API calls for six months of pytest.
- **No future leakage.** Every query is evaluated *as of* the event time:
  - Labels are rebuilt from the label events.
  - Later comments, reviews and re-run attempts are hidden. Inline comments of a pending review appear only
    once that review is submitted.
  - Workspaces are `git clone --shared` checkouts whose only refs are the commit under test and `main` *as it
    was*. The main commit is found with `--first-parent`, because merged PR commits carry earlier dates.

  Each of these cases was a real leak found while building this, and each has a regression test in
  `tests/test_replay.py`. One known remaining channel: unreferenced future objects are still in the shared
  object store, so an agent that deliberately enumerates all objects could find them.
- **Sandbox commands.** `hist` (issue, pr, ci, log, search, recent, labels, commits) and `ci-run` (the
  project's own test-suite in one small shared venv; pytest's `testing/` runs locally, about 213 tests in 12s).
- **Tasks with ground truth from what happened later** (`ospm replay tasks`):
  - **Triage:** kind from maintainer labels or from the changelog fragment that eventually fixed the issue
    (`changelog/<issue>.bugfix.rst` means bug). Decision from close reason, "Duplicate of #N" comments, and
    `status: needs information` / `invalid` labels.
  - **Review:** the verdict of the first maintainer review, plus the file:line of its inline comments, on the
    exact commit reviewed.
  - **CI:** "caused by this change" vs. "not caused by it". Not caused means it passed on a re-run, or the same
    failure hit other branches or main at the same time. Also the failing tests (ANSI-stripped log) and a
    category.

## 4. Team-GEPA (`ospm/evolution/team_gepa.py`)

Kept from GEPA (Agrawal et al., 2025): reflective mutation from execution traces and natural-language feedback,
per-instance Pareto-frontier parent sampling, a strict minibatch acceptance gate followed by full Pareto-set
evaluation, and system-aware merge.

New for multi-agent systems:

1. **Blame-directed module selection.** The reflector reads the *hierarchical* trajectory (orchestrator steps
   with the specialist transcripts nested inside) and the replay-grounded feedback. It must name the one role
   responsible, separating coordination failures from specialist failures, and edit only that role.
   `--no-blame` restores GEPA's round-robin over the roles that ran, as an ablation.
2. **Invocation-aware caching.** An episode depends only on the genomes of roles that actually ran, because role
   descriptions live in the orchestrator's genome. Cache keys are *(task, {role: hash} of the roles that ran)*,
   so mutating the security agent re-runs only episodes where security was spawned.
3. **Cost-aware Pareto front.** A candidate is dominated only if another is at least as good on every task and
   no more expensive on average.
4. **Replay-grounded feedback.** The feedback text says what really happened: the maintainer's labels and close
   reason, their inline comments at path:line, whether the job passed on re-run.
5. **Placeholder-safe rewriting.** Rewritten prompts are repaired so tool docs, the output contract and the
   playbook slot survive.

## 5. Commands

```bash
ospm mirror pytest-dev/pytest --since 2026-04-01 --out data/pytest      # ~18 min, ~2k API calls
ospm replay setup-ci --data data/pytest                                   # small shared venv for ci-run
ospm replay tasks --data data/pytest                                      # mine tasks + ground truth
ospm replay stats --data data/pytest                                      # counts and disk usage
ospm --llm openai team run  --data data/pytest --task triage:14953        # one episode (full team)
ospm --llm openai team eval --data data/pytest --entry orchestrator --split val --limit 12
ospm --llm openai team eval --data data/pytest --entry specialist   --split val --limit 12
ospm --llm openai --budget 3 team evolve --data data/pytest --iterations 10 --minibatch 4
ospm trace data/pytest/traces/<file>.json.gz                              # inspect a trajectory
ospm replay clean --data data/pytest [--traces]                           # free disk space
```
