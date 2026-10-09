# ospm — a self-evolving OSS project-manager agent

`ospm` is a multi-agent project manager for open-source repositories. It triages issues, grooms and plans the
backlog, diagnoses CI failures, reviews pull requests for project conventions and security problems, and
proposes releases. It also improves itself over time through **non-parametric evolution**: the LLM's weights
never change. What changes is each agent's *genome*: its natural-language policy, a playbook of executable
rules, retrieval and scoring parameters, and pinned few-shot examples. Evolution is driven by maintainer
feedback and a **GEPA**-style reflective search.

```
                         ┌───────────────────────── persistent memory (SQLite) ─────────────────────────┐
                         │ episodes · labelled cases · genomes + lineage · promotions · rule credit ·    │
                         │ LLM response cache                                                           │
                         └──────▲──────────────────────▲────────────────────────────▲───────────────────┘
   new issue ──► TriageAgent ───┤  backlog ──► BacklogPlanner ──► milestones ──► release planner
                 (valid? scope? kind? priority? duplicate?)          │
   pull request ─┬► CITriageAgent  (per failed job: category, root cause, fix, retry?)
                 ├► ReviewAgent    (project conventions, file/line comments)       ──► merge-readiness verdict
                 └► SecurityAgent  (injection, secrets, TLS, deserialisation, …)
   maintainer feedback (labels, /ospm commands) ──► memory ──► GEPA evolution ──► promotion gate ──► live genome
```

## Agent team on replayed real history (new)

Tool-using agents built like mini-swe-agent v2, an orchestrator that delegates to specialists, a leak-proof replay environment built from pytest-dev/pytest's real history, and **Team-GEPA** (blame-directed, invocation-cached, cost-aware GEPA). See [`docs/AGENT_TEAM.md`](docs/AGENT_TEAM.md).

## Why this design

| Requirement from the brief | Where it lives |
|---|---|
| Validate new issues and check scope | `TriageAgent` (`ospm/agents/triage.py`): kind / decision (accept, needs-info, out-of-scope, duplicate, spam) / priority, duplicate detection against the tracker, a reply to the reporter, suggested labels |
| Backlog prioritisation, grooming, assignment | `BacklogPlanner` (`ospm/agents/planning.py`): explainable score (priority, reactions, age, roadmap fit, regressions), readiness, owners from components, capacity-bounded milestones |
| Help contributors whose PRs fail CI | `CITriageAgent` (`ospm/agents/ci_triage.py`): salient-line extraction across ecosystems (pytest, go, jest, cargo, gradle, docker, npm, scanners), failing tests, root-cause line, concrete fix, re-run advice for flaky infra |
| Project conventions and review guidelines | `ReviewAgent` (`ospm/agents/diff_agents.py`): conventions from the project file, findings located to `file:line` |
| Security review | `SecurityAgent`: secrets, SQL/command injection, unsafe deserialisation, disabled TLS, weak crypto, path traversal, eval/exec |
| Roadmap and release scheduling | `plan_release`: semver bump from conventional commits/labels (pre-1.0 aware), grouped release notes |
| Specialised agents + shared memory | `ProjectManager` (`ospm/manager.py`) + `Memory` (`ospm/memory.py`) |
| Evaluation metrics | `ospm/evolution/metrics.py`, labelled benchmarks in `ospm/benchmarks/` |
| Self-improvement (non-parametric evolution) | `ospm/evolution/` — GEPA engine, reflective operators, promotion gate |

## How an agent decides

Every agent is **hybrid**, and its genome configures both modes:

1. **Heuristic policy** (free, deterministic, always available): playbook rules + k-NN votes from labelled past
   cases (TF-IDF retrieval) + structural signals (log structure, diff paths, link counts, …).
2. **LLM policy** (optional): the same playbook, the retrieved cases, pinned exemplars and the heuristic
   pre-assessment go into a schema-constrained LLM call. If the call fails or the budget is spent, the agent
   falls back to the heuristic decision instead of failing.

Because playbook rules are both executable and rendered into the prompt, one learned lesson improves both modes.

## Two learning loops

**Online (immediate, no search).** Each decision is stored as an episode. When a maintainer corrects or confirms
it (adds a label, or comments `/ospm triage priority=P1`), the labelled case enters retrieval memory, so the next
similar item benefits right away. The rules that fired get credit or blame. In the demo's cold-start stream of
96 issues, mean triage score rises from **0.61 → 0.67 → 0.80** over successive thirds using memory alone.

**Offline: GEPA evolution** (`ospm/evolution/gepa.py`), after Agrawal et al., *GEPA: Reflective Prompt Evolution
Can Outperform Reinforcement Learning*:

* **Reflective mutation.** Run the parent on a minibatch of training cases and attach natural-language feedback
  (what was wrong, which rules fired). Then mutate one *module* of the genome, round-robin per candidate:
  * `reflection`: an LLM diagnoses the failure patterns and rewrites the policy, adding, removing and re-weighting
    rules (validated before use; invalid regexes are dropped);
  * `induction`: an offline analyst induces precise phrase rules (precision, support and *lift over base
    rate*) from failures, and vetoes or demotes rules that misfired;
  * `params`: bounded jitter of retrieval and scoring knobs;
  * `exemplars`: re-pins few-shot cases covering the labels the agent gets wrong.
* **Pareto-based selection.** Each candidate keeps per-instance scores on the Pareto (val) set. Parents are
  sampled from the non-dominated candidates, weighted by how many instances they are best on, so the search
  keeps diverse specialists instead of collapsing onto one average.
* **System-aware merge.** For two Pareto candidates with a common ancestor, take each module from the side that
  evolved it (playbooks are unioned when both changed).
* **Staged evaluation (FlashEvolve-style acceleration).** Content-addressed genomes and a `(genome, example)`
  decision cache mean no genome is re-run on the same case. LLM responses are cached persistently, evaluation is
  parallel, and a gate runs minibatch → full train (no regression, heuristic mode) → Pareto set. Most bad
  children are rejected cheaply.
* **Promotion gate.** The best candidate goes live only if it beats the *currently deployed* genome on the
  Pareto set by `min_delta` without regressing on train. The held-out **test** split is reported and never
  used for any decision. Every run, candidate, lineage edge and promotion is recorded.

Splits follow GEPA's protocol: *train* = feedback and minibatch source (and the k-NN memory), *val* = Pareto and
selection set, *test* = held out.

## Results

Benchmarks: the bundled fictional project **inkwell** (a static-site generator) with hand-labelled data:
96 issues, 48 CI logs across 8 categories, and 40 pull requests with review and security labels. Splits are
stratified 60/20/20.

**Heuristic policy, offline GEPA (no LLM, ~1 s per run), mean over seeds 0–2, 40 iterations:**

| agent | val: seed → evolved | **test: seed → evolved** |
|---|---|---|
| CI triage | 0.60 → 0.70 | **0.78 → 0.89** (all seeds) |
| issue triage | 0.72 → 0.74 | **0.77 → 0.78** (never below seed) |
| review / security | unchanged | unchanged: these need regex-level lessons, which LLM reflection provides |

**LLM policy (`gpt-6-luna`), seed genomes** (224 calls, $0.07 for all four agents × all splits):

| agent | train | val | test |
|---|---:|---:|---:|
| issue triage | 0.936 | 0.932 | 0.947 |
| CI triage | 1.000 | 1.000 | 1.000 |
| review | 1.000 | 1.000 | 1.000 |
| security | 0.958 | 0.875 | 1.000 |

**GEPA with GPT-6 Luna as the reflector, evolving the *free* heuristic policy** (24 iterations each):

| agent | val: seed → evolved | **test: seed → evolved** | reflection cost |
|---|---|---|---:|
| review | 0.75 → 0.83 | **0.71 → 0.875** | $0.010 |
| security | 0.75 → 0.875 | 0.50 → 0.50 (8-item test set; no transfer) | $0.012 |

Here the LLM writes regex-level playbook rules (mutable defaults, missing annotations, deserialisation sinks)
that the zero-cost policy then runs with no LLM in the loop.

**GEPA with GPT-6 Luna as both reflector and policy** (triage, 24 iterations, 300 calls, $0.10): several
reflections improved minibatches, but none beat the incumbent on the Pareto set (0.932 at its ceiling), so the
gate kept the seed (test 0.947). With a strong policy model this benchmark is close to saturated. Gains need
harder or larger data, which is what real repository feedback provides.

Two genuine defects were found *by* live evaluation and fixed (each with a regression test): relational labels
("duplicate") leaking into the LLM prompt via retrieved neighbours, which made the LLM mark *originals* as
duplicates (triage val 0.910 → 0.932 after the fix); and a crash in prompt rendering for multi-label neighbours.
Two inconsistencies the model pointed out were also corrected: a benchmark PR used AWS's documented placeholder key as its "leaked secret", and the project file's type-hint convention did not say it applies only to library code (as the labels assumed).

Small benchmarks make val and test noisy (8–19 items). Treat single-run differences of one item as noise. That is
why the promotion gate exists and test is reported separately.

## Quick start

```bash
pip install -e ".[all]"          # Python ≥ 3.11; the core has no third-party dependencies
pytest                           # 80 offline tests (live API tests: pytest -m live, 4 tests)

ospm demo                        # whole pipeline + both learning loops on 'inkwell', offline, in seconds
ospm --llm openai demo           # same, with GPT-6 Luna as policy/reflector (costs cents)
```

Everyday commands (add `--llm openai|anthropic` to use an LLM, `--json` for machine output):

```bash
ospm triage --title "Crash on serve" --body "Traceback ..."        # or --github owner/repo#123 [--apply]
ospm ci --log failed-job.log
ospm review --diff change.diff                                      # or --github owner/repo#45 [--apply]
ospm backlog --github owner/repo --milestones v1.1,v1.2
ospm release --github owner/repo                                    # next version + release notes
ospm feedback --agent triage --item issue-123 --set decision=duplicate priority=P3
ospm eval --agent all --split all
ospm evolve --agent triage --iterations 30 [--reflection openai] [--policy-llm]
ospm memory stats | genome triage | promotions | runs | lineage RUN_ID | rules triage
```

GitHub writes are **dry-run unless `--apply`** is given.

## Using it on your repository

1. Copy `ospm/benchmarks/inkwell.toml` and describe your project: scope, out-of-scope keywords,
   components with paths and owners, conventions, roadmap, and maintainers. Maintainers' labels and `/ospm`
   commands count as ground truth.
2. Add `.github/workflows/ospm.yml` (included). It triages new issues, reviews PRs (`pull_request_target`,
   never executing PR code), explains failed CI runs, learns from maintainer labels and `/ospm` comments, and
   runs evolution weekly. Memory persists through the Actions cache.
3. Projects other than inkwell learn only from their own feedback. `evolve` waits until there are labelled
   cases in both train and val (about 5+).

## Configuration and cost control

| env var | meaning |
|---|---|
| `OSPM_LLM` | `none` (default), `openai`, `anthropic`, `auto` |
| `OSPM_MODEL` | defaults: `gpt-6-luna` (OpenAI), `claude-opus-5-5` (Anthropic) |
| `OSPM_BUDGET_USD` | **hard per-process spend cap** (default 2.00). Checked before every call; agents fall back to heuristics when it is reached |
| `OSPM_DB`, `OSPM_PROJECT` | memory file, project TOML |
| `GITHUB_TOKEN` / `GH_TOKEN` | GitHub API (needed for CI job logs and writes) |

Backends use structured outputs: OpenAI's Responses API with a strict JSON schema, and Anthropic's Messages API
with `output_config.format`, cached system prompts and server-side refusal fallbacks (`fallbacks: "default"`).
Every response is validated against the schema again locally. Both wire formats are tested against the real
SDKs through a mocked transport (`tests/test_llm.py`).

## Layout

```
ospm/
  agents/        base (hybrid policy), triage, ci_triage, diff_agents (review, security), planning
  evolution/     gepa (engine), operators (reflection, induction, params, exemplars, merge), metrics
  benchmarks/    labelled data (issues, CI logs, PRs), inkwell.toml, stratified splits
  llm/           provider-neutral interface, OpenAI + Anthropic backends, fake, budget meter
  integrations/  GitHub REST client (dry-run by default), event handling and feedback parsing
  manager.py     the SDLC orchestrator      memory.py   SQLite memory     genome.py  genomes and rules
  cli.py · render.py · demo.py
tests/           offline suite + opt-in live tests
```

## Capstone milestones this supports

1. **Single-purpose agent**: run `TriageAgent` alone (`ospm triage`, `ospm eval --agent triage`).
2. **Specialised agents + orchestration**: `ProjectManager.review_pr` combines CI triage, review and security
   into one verdict.
3. **Shared memory + metrics**: episodes, cases and rule credit; per-field accuracy and macro/micro-F1.
4. **Self-improvement**: GEPA over instructions, playbook, params and exemplars, with lineage and a promotion gate.
5. **Research extensions**: other selection schemes (MAP-Elites), cost-aware multi-objective Pareto
   (score vs. tokens), RL-tuned operator scheduling, adversarial-testing and soak-test agents, and cross-project
   transfer of playbooks.

## Limitations

* Benchmarks are small and synthetic (but hand-written and checked against their labelling definitions in
  tests). Larger, real repository histories are the next step.
* Offline induction learns lexical rules only. Structural patterns (regexes) come from LLM reflection.
* The backlog score and release planner are transparent heuristics; they are not learned.

## References

* *Position: Agentic Evolution is the Path to Evolving LLMs*
* Agrawal et al., *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning* (2025)
* *FlashEvolve: Accelerating Agent Self-Evolution*
* Novikov et al., *AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms* (2025)
