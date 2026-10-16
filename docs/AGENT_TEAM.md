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

- **Context compaction.** On OpenAI the provider compacts server-side. Otherwise, when the context passes
  `compaction_tokens`, the summary is requested with `Conversation.side_query`: the agent's own request (same
  system prompt, tools and history) plus a trailing `COMPACTION_INSTRUCTION` user message, after which the
  conversation is left unchanged. Because the request extends the last call's prefix, the provider serves that
  prefix from its cache, as in DeepSeek Harness; a separate summariser prompt would re-read the whole history at full
  price. If the model calls a tool instead of answering, a separate summariser call is the fallback. The whole
  history is then replaced by one message holding the task, the summary and the last observation; nothing older is
  replayed. This "simple compaction" is the shape Anthropic recommends for client-side compaction with preserved
  thinking. It never happens mid tool round, and it is equally valid for OpenAI's stateless Responses input. The
  summary call's tokens and cost are recorded on the `compaction` step.
- **Staged elision (team episodes).** Following Fan et al. (2026), stale tool output is cut before any summary.
  In a team episode, once an agent's context passes 64k tokens (`TEAM_ELIDE_TOKENS`, `ospm/team/runner.py`), every
  tool result except the last 6 is replaced by a one-line stub saying how long it was and to rerun the command if
  needed (`Conversation.elide`), and an `elision` step is recorded. Results under 1,200 characters are kept.
  Summarising comes only after that. The paper found re-fetching cut output almost never helped, so the stub is
  final. Single-agent episodes never elide: their contexts stay around 10k tokens.
- **Stuck detection.** Also from Fan et al. When an agent issues the same tool calls 5 steps in a row, it gets a
  note that the output will not change (a `stuck` step). After 8 identical steps that all failed (non-zero exit,
  timeout or tool error), it gets the step limit's final turn to submit what it has (`LimitsExceeded: stuck` if it
  does not). This change and elision came after most cached episodes were recorded. They change an episode only
  when it loops or its context grows past 64k, so the episode cache version was not bumped.
- **Stable prompt parts first.** Providers cache the longest identical prefix of a request, so the system prompts
  put what evolution changes last: the orchestrator's team listing and every role's playbook come after the fixed
  instructions, tool documentation and output contracts. Editing an agent type's description now keeps 87% of the
  orchestrator prompt as a cached prefix, against 11% before. In our traces 79–89% of input tokens were already cache
  reads, because each conversation only grows at the end.
- **Truncated turns are resampled.** A turn that runs out of output tokens (usually runaway reasoning) raises
  `TruncatedTurn` and is sampled again, up to `TRUNCATION_RETRIES` (2) times, instead of ending the episode
  (`ospm/harness/models.py`).
- **Tool timings.** Each bash or tool call's wall time is stored in its trace result (`"s"`), for derived latency
  (section 2).
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

**Beyond review.** `sdlc_library()` is `seed_library()` plus types for the rest of the maintainer workflow:
`planner` (`plan` contract: approach, files, steps, risks), `implementer` (`patch`: status, changed files, tests run),
`test_writer` (`testcases`: tests added, whether they fail before and pass after) and `release` (`release`: version
bump, highlights, breaking changes, notes). Its orchestrator also accepts `implementation` events (the change in the
shared checkout is the deliverable; `patch` is required) and `release` events. Writers are told to touch only the files
their brief assigns and never to stash, check out, reset or commit, since teammates share the checkout. These
contracts live in `SDLC_CONTRACTS`, apart from `CONTRACTS`, so the seed teams' prompts and ids are unchanged. No replay
task scores the new types yet.

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
- **Notepads** (`TeamGenome.board`, off by default and part of the genome id). Agents share notes without sending
  messages:
  - **Pads.** A `global` notepad is seen by every agent of the episode. The orchestrator also assigns task notepads
    with `spawn_agent(..., pad="fix")`, and a task notepad is seen only by the agents given that name and by the
    orchestrator, which sees every pad.
  - **Tools.** `post_note(kind, note, paths, pad)` takes kind `covered`, `finding`, `working_on` or `question`. A
    specialist posts to its own pad by default or to `global`, and only the orchestrator may post to another pad.
    `read_notes(all)` returns the visible notes from other agents since the caller last read.
  - **Rules.** Notes are append-only and attributed (pad, agent id, role, time), with at most 800 characters and 30
    notes per agent. They live only for the episode and are returned as `Episode.board`. The prompt tells agents that
    notes are claims to check, not instructions.
  - **Evolution.** Team evolution can switch the notepads on or off with a team-level `set_board` edit. `--board`
    starts the seed library with them on.
- **Task board** (`TeamGenome.task_board`, off by default and part of the genome id). Agents share a board of tasks
  with owners, dependencies and write scopes, kept in memory for the episode:
  - **Tools.** Every agent gets `task_list()`, `task_update(task_id, revision, status, owner, note, write_scope)`
    and, for the orchestrator only, `task_create(title, description, depends_on, write_scope, owner)`. Status is
    `pending`, `in_progress`, `blocked`, `done` or `cancelled`.
  - **Revision checks.** An update succeeds only if `revision` is the task's current one; a stale update returns the
    current task so the caller can re-read and retry. Each success bumps the revision and records `updated_by`.
  - **Ownership and dependencies.** The orchestrator may update any task. Another agent may update a task it owns or
    claim an unowned one with `owner="me"`. A task cannot become `in_progress` or `done` while a dependency is not
    `done`.
  - **Advisory write scopes.** A scope is a list of files or directories. When a task goes `in_progress` and its scope
    overlaps (same path or one inside the other) that of another owner's in-progress task, the update still succeeds
    but returns `warnings`; `task_list` also returns the current `conflicts`. Nothing blocks writes.
  - **Limits.** At most 30 tasks, 20 notes per task (500 characters each) and 20 paths per scope. Tasks are returned
    as `Episode.tasks`.

**Derived latency** (`ospm/harness/latency.py`, after DeepSeek Harness's Agent Teams). An episode's latency is the
critical path through its agents, computed from the trace rather than the wall clock, which rate-limit retries,
concurrent episodes and provider queueing distort:
- **Model time.** A step costs new prompt tokens / 4,000 per second, plus output tokens / 80 per second, plus 0.5 s.
  Only tokens added since the previous step count, assuming a warm prefix cache. These rates are nominal; only ratios
  between systems matter.
- **Tool time.** Bash and tool calls cost their measured duration, or 1 s each for traces recorded before durations
  were. Coordination calls are instant.
- **Dependencies.**
  - A child starts after the model turn that spawned it, and its steps run in sequence.
  - A parent cannot continue past a `finished` notification, whether from `wait_for_agents` or auto-delivered, before
    that child's derived finish. Parallel specialists therefore overlap, and waiting on them one by one adds up.
  - Approximation: a resumed child continues its own timeline.

On the SGLang traces the nominal rates match the real wall clock. The single reviewer has a median of 78 s derived
against 79 s wall. The team has 126 s against 138 s, for about 3× the steps and cost: its parallelism hides most of
its extra work.

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
    The clone's `.git/logs` is removed: its HEAD reflog named the mirror's current (future) default-branch tip.

  Each of these cases was a real leak found while building this, and each has a regression test in
  `tests/test_replay.py`. One known remaining channel: unreferenced future objects are still in the shared
  object store, so an agent that deliberately enumerates all objects could find them.
- **Large repositories.** Checkouts are pooled and reused (`ReplayRepo.pool_size`); each reuse gets a brand-new
  `.git` and a forced checkout and clean, so nothing of the earlier episode survives. Blobs missing from a
  partial-clone mirror are fetched by object id before checkout (`prefetch`; 4 s instead of about 150 s per new
  Home Assistant commit). Recursive greps run through `git grep` (`ospm/harness/fastgrep.py`).
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
