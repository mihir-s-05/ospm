# ospm: a self-evolving OSS project-manager agent

`ospm` is a multi-agent project manager for open-source repositories. It triages issues, diagnoses CI failures,
reviews pull requests and plans releases. It also improves itself without retraining: what evolves is its
prompts, rules, skills, agent types and team coordination, never the model's weights.

To test that honestly, `ospm` includes a **replay environment** built from real repository history. An agent
handles a real event (an issue opened, a PR ready for review, a CI job failing) exactly as the repository was at
that moment, and is scored against what the maintainers actually did.

## Principles

1. **Reality is the benchmark.** Ground truth comes from maintainers' real decisions, not labels we wrote.
2. **No peeking at the future.** Every view of the repository is frozen at the event time. Each leak we found
   (later comments, commits pushed after a review, future git refs) has a regression test.
3. **Improve the system, not the weights.** Changes are readable text (rules, skills, agent descriptions), so
   they are cheap, auditable and reversible.
4. **Nothing is adopted without independent evidence.** A learned change is kept only if replaying *other*,
   earlier PRs shows it helps, without new false alarms or (for teams) extra cost.
5. **Measure the measurement.** The scoring rule was itself validated against hand-labelled data, and replaced
   when it turned out to be wrong.

## What is in it

| part | what it does | where |
|---|---|---|
| Replay environment | Mirrors a repository's issues, PRs, reviews, CI runs and logs. Builds time-frozen workspaces and tasks with maintainer ground truth (triage, review, CI, review concerns). Handles PyTorch's ghstack and bot-landed PRs. | `ospm/replay/` |
| `hist` | The agents' command-line view of history as of the event: issues, PRs and discussions, earlier maintainer review comments on a file (`hist reviews --path`), CI runs and logs, commits. | `ospm/replay/hist.py` |
| Agent harness | mini-swe-agent-style agents (bash, git, local test runs) with a sandbox, context compaction and hierarchical traces. | `ospm/harness/` |
| Agent team | An orchestrator runs specialists in the background, like Claude Code's subagents: spawn, message, wait, close, resume. It sees only each specialist's final message and work report. The specialists form an evolvable **library of agent types**. | `ospm/team/` |
| Learning | GEPA and Team-GEPA prompt evolution; validated scoped lessons; **skills** evolved by post-mortems of traces; review **rubrics** mined from recurring maintainer concerns and evolved by post-mortems behind a sampled replay gate; a ceiling test; **team evolution** of agent types, the library and coordination rules, with per-agent credit and cost-aware acceptance. | `ospm/evolution/`, `ospm/learning/` |
| Semantic judge | Decides whether an agent's comment raises the same concern as a maintainer's, using TypeSafe's Jev (a calibrated true/false model). | `ospm/replay/semantic.py` |

Design details: [`docs/AGENT_TEAM.md`](docs/AGENT_TEAM.md) (harness, team, replay),
[`docs/LEARNING.md`](docs/LEARNING.md) (tasks, methods, all results),
[`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md) (where interventions could have helped).

## Results

All agent runs use **GPT-6 Luna**. The main task replays a maintainer's review of a PR: the agent reviews the exact
commit the maintainer saw and is scored on whether it raises the maintainer's concerns. On PRs maintainers approved
as-is, it is scored on staying silent. Splits are chronological: learn on older PRs, test on newer ones.

**1. Measurement.** The original match rule ("same file, within 5 lines") was checked against 104 hand-labelled
pairs:

| judge | AUC | accuracy | precision | recall |
|---|---:|---:|---:|---:|
| ±5-line rule | 0.78 | 0.83 | 0.59 | 0.70 |
| **Jev** (TypeSafe; used now, p ≥ 0.8) | **0.999** | **0.99** | **1.00** | **0.96** |
| OpenAI Decisions API (gpt-6-luna, p ≥ 0.5) | 1.000 | 0.97 | 1.00 | 0.87 |

Jev is now the default scorer (`--match semantic`), and every run below was rescored with it. It costs about $0.03
per thousand judgements.

**2. What the reviewer catches** (semantic recall of maintainer concerns on the test PRs):

| repository | plain reviewer | + memory | + lessons | + `hist reviews` tool | + evolved skills | + mined rubric | + evolved rubric | team (orchestrator + specialists) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vLLM | 0.00–0.02 | 0.00–0.04 | = plain | | | | | |
| SGLang | 0.10 | 0.09 | = plain | 0.13 | 0.13 | 0.04 | | 0.08 (0.12 with the tool) |
| PyTorch | 0.01 | 0.00 | | 0.00 | 0.00 | | | |
| Django | 0.09 | | | | | 0.04 | = mined | |
| Home Assistant | 0.03 | 0.00 | | 0.00 | 0.00 | 0.06 | 0.04–0.09 | |

Home Assistant figures are on 192 test PRs (the memory, tool and skills arms on the first 96). Every paired
comparison carries an exact McNemar p-value, Benjamini–Hochberg-corrected across the systems in a run
(`ospm/learning/stats.py`).

- **No learned method has clearly beaten the plain reviewer yet.** Validated lessons: 0 of 57 adopted on vLLM and SGLang (2 on
  pytest, which never applied to a test PR). Evolved skills: 1 of 31 adopted on SGLang, 1 of 20 on PyTorch, 1 of
  28 on Home Assistant. Every other difference is within noise.
- **The team matches one reviewer at about twice the cost.** Traces show coordination loses nothing: every
  concern a specialist found reached the final answer. But it adds nothing either. The orchestrator re-read the
  diff itself, rarely followed up, and spawned a security agent that never commented.
- **What gets found** is concrete bugs: 29% of correctness concerns, pooled over four runs. Design objections
  and test gaps were never found. 52–63% of maintainer concerns on vLLM, SGLang and PyTorch need knowledge beyond
  the diff.
- **Evolving the team cuts its cost, not its misses.** On SGLang, team evolution adopted 3 of 12 edits: a
  coordination rule (*"Do not spawn security by default; spawn it for security-sensitive changes"*) and removing the
  `tests` and `history` types, which never found a concern alone. On 32 test PRs the evolved team has the seed
  team's recall (0.13) at 24% lower cost ($0.035 vs $0.047 per PR), with slightly more noise on clean PRs (paired
  2/6/24).

- **Why: maintainer concerns rarely repeat a project rule.** A pairwise judge of each new concern against earlier
  ones from other PRs (`ospm/replay/recurrence.py`) finds that only 2–6% restate a project-specific rule on vLLM,
  SGLang, PyTorch, pytest, Django and Rust. A longer mirror adds a few points. **Home Assistant is the exception**
  at 13% (25% loose): its translation, test and quality-scale conventions are applied across integrations. The
  agent-tool repositories (codex, opencode, pi) have almost no public human review to replay.
- **Post-mortems miss those conventions.** On 96 Home Assistant test PRs, the
  plain, memory, tool and evolved-skill reviewers all had semantic recall ~0. The reviewer wrote 0.27 comments per
  PR with maintainer concerns and approved most. Of 28 evolved skill edits, 1 generic one was adopted, because the
  post-mortems proposed integration-specific skills.
- **A mined rubric finds some of them, on Home Assistant only.** `ospm/learning/rubric.py` mines the rules
  directly:
  - it clusters earlier concerns that the recurrence judge says apply the same rule;
  - it writes them as a checklist: 58 items in 6 areas, for one LLM call ($0.01);
  - the reviewer gets the items for the files a PR touches inlined into its context.

  On 192 Home Assistant test PRs recall went from 0.026 to 0.061 (e.g. *"move the dependency bump into a separate
  PR"*): 4 wins and 0 losses on PRs with concerns (p = 0.13). Offered as on-demand skills instead, the same rubric
  did worse. On Django and SGLang, where only 2–6% of concerns recur, the rubric made the reviewer comment more but
  match less (Django 5/5, SGLang 3/8).
- **Rubric items evolved by post-mortems do not reliably transfer.** Post-mortems of the training PRs add, edit or
  remove one item at a time (`RubricLearner`). An edit is kept only if replaying earlier PRs whose concerns it
  applies (found by a Jev judge over all of history) raises those concerns without new comments on clean PRs.
  - The first run adopted 2 of 28 edits and scored 9 wins, 0 losses on PRs with concerns (p = 0.004).
  - A stronger gate (2 epochs, 3 samples per validation PR and arm, at least 2 supporting PRs, a memory of rejected
    edits) adopted 4 of 42, nearly a superset of the first two. It scored 3/1 (p = 0.63), below the mined rubric
    it started from.
  - So the 9/0 was most likely a lucky draw: in validation, the same PR and context raised its target concern in
    0 to 3 of 3 runs. Validation PRs state the item almost verbatim, so passing it says little about a new PR.
- **The reviewer is not the bottleneck; the rubric's coverage is.** On the 96 Home Assistant test PRs with
  concerns, each concern was rewritten as the general rule it applies (`ospm/learning/ceiling.py`). With those
  rules as rubric items, recall is 0.30 (35/0); told the PR violates them, 0.67 (69/0). The learned rubrics reach
  0.04–0.09.

**3. Data audit** (pytest, pip, pydantic). Defects that surface after merge are rare (about 2% of PRs), and their
evidence almost always existed at review time. Hence the focus on review.

**4. Synthetic benchmark** (`inkwell`, a hand-labelled fictional project: 96 issues, 48 CI logs, 40 PRs). It was
the first test bed for the agents and GEPA.
- **Rule-based agents improved under evolution.** CI-triage test score went from 0.78 to 0.89, and review from 0.71
  to 0.875.
- **LLM agents were already at 0.93–1.0**, so the benchmark saturated. That is why we moved to real history.

All experiments so far cost about $51 of API usage.

## Quick start

```bash
pip install -e ".[all]"   # Python >= 3.11
pytest                    # 257 offline tests (live API tests: pytest -m live)
ospm demo                 # agents + both learning loops on the synthetic project, offline
```

Replay a real repository (needs `GITHUB_TOKEN`; `OPENAI_API_KEY` for agents; `TYPESAFE_API_KEY` for the judge):

```bash
ospm mirror sgl-project/sglang --since 2026-06-01 --out data/sglang --blobless   # history + bare clone
ospm replay tasks --data data/sglang                                              # tasks + ground truth
python -m ospm.replay.concern_labels --data data/sglang --batch                   # concern taxonomy (Batch API)
python -m ospm.replay.recurrence --data data/sglang --pool 1000                  # do maintainer concerns recur?

# single reviewer: plain / memory / tool / evolved skills, learned on older PRs, tested on newer ones
python -m ospm.learning.experiment --data data/sglang --train 64 --test 48 \
    --systems none,memory,tool,skills --budget 6
# the team: agent library + team evolution
python -m ospm.learning.team_evolution --data data/sglang --train 32 --test 32 --budget 9
# mine a review rubric from recurring concerns; evaluate it as mined (rubric) and evolved by post-mortems
python -m ospm.learning.rubric --data data/home-assistant --out data/results/rubric_ha
python -m ospm.learning.experiment --data data/home-assistant --train 64 --test 192     --systems none,rubric,evolved_rubric --library data/results/rubric_ha_skills.json     --epochs 2 --samples 3 --min-support 2 --min-hit-gain 2 --max-fp-increase 0.25 --learn-budget 5.5 --budget 7
# ceiling: the reviewer given each test concern's rule as a rubric item, or told the PR violates it
python -m ospm.learning.ceiling --data data/home-assistant --library data/results/rubric_ha_skills.json     --baseline data/results/learning_ha_rubric2.json --out data/results/ceiling_ha.json
# rescore a finished run with the semantic judge (no new episodes)
python -m ospm.learning.rescore --data data/sglang --result data/results/learning_sglang_skills.json
```

Everyday commands (add `--llm openai|anthropic`, `--json` for machine output):

```bash
ospm triage --title "Crash on serve" --body "..."     # or --github owner/repo#123 [--apply]
ospm ci --log failed-job.log
ospm review --diff change.diff                         # or --github owner/repo#45 [--apply]
ospm release --github owner/repo
ospm evolve --agent triage --iterations 30 [--reflection openai]
```

GitHub writes are dry-run unless `--apply` is given. `.github/workflows/ospm.yml` runs triage, review and CI
explanations on a repository and learns from maintainers' labels and `/ospm` comments.

## Cost control

| env var | meaning |
|---|---|
| `OSPM_LLM`, `OSPM_MODEL` | provider (`none`, `openai`, `anthropic`, `auto`) and model (default `gpt-6-luna`) |
| `OSPM_BUDGET_USD` / `--budget` | hard per-process spend cap, checked before every call |
| `OPENAI_API_KEY`, `TYPESAFE_API_KEY`, `GITHUB_TOKEN` | agents, semantic judge, mirroring |

Episodes, judgements and reflector answers are cached on disk, so interrupted runs resume without paying twice.
Rate-limited episodes wait and rerun instead of failing the experiment.

## Layout

```
ospm/
  replay/     mirror, time-frozen store and workspaces, hist CLI, tasks, scoring, semantic judge, ghstack
  harness/    mini-swe-agent-style agent loop, sandboxed environment, provider wire formats
  team/       roles and the agent-type library, asynchronous team runner
  learning/   experiments: interventions, skills + post-mortem, mined rubrics, team evolution, rescoring
  evolution/  GEPA and Team-GEPA
  agents/ llm/ integrations/ benchmarks/   original hybrid agents, LLM backends, GitHub client, synthetic data
tests/        offline suite (fake models and stubbed judges)
docs/         design notes and full results
```

## References

- Agrawal et al., *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning* (2025)
- *mini-swe-agent* (SWE-agent team): the harness design
- *Agentic Context Engineering* (ACE): the memory baseline
- Śliwerski et al., *When do changes induce fixes?* (SZZ), used in the data audit
- *Position: Agentic Evolution is the Path to Evolving LLMs*; *FlashEvolve*; Novikov et al., *AlphaEvolve* (2025)

## License

MIT, see [`LICENSE`](LICENSE).
