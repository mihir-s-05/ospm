# Learning where to intervene: review-concern tasks and validated interventions

This builds on the findings in [`DATA_AUDIT.md`](DATA_AUDIT.md). Maintainer review corrections are the densest
cross-stage signal, almost all detectable problems had their evidence at PR review, and validation must use PRs
other than the one a lesson came from.

## Target repository: vLLM

The repositories were surveyed with identical metrics over 90 days (`python -m ospm.replay.survey`, raw numbers
in `data/repo_survey/survey.json`):

| repo | human PRs opened / merged | reverts | follow-up fixes | regression issues | merged PRs with maintainer inline review | CI logs via API |
|---|---|---:|---:|---:|---:|---|
| **vllm-project/vllm** | 9,839 / 3,674 | 55 | 317 | 403 | 17% (~600 PRs) | no (Buildkite logs need a login) |
| sgl-project/sglang | 10,941 / 4,514 | 38 | 296 | 194 | 10% | yes (GitHub Actions; 12/12 sampled heads failing) |
| pytest-dev/pytest | 310 / 76 | 0 | 5 | 12 | 27% | yes |
| pydantic/pydantic | 366 / 186 | 4 | 6 | 18 | 10% | yes |
| pypa/pip | 124 / 48 | 1 | 1 | 2 | 27% | yes |

vLLM has 10–30× the cross-stage signal of the earlier candidates, mostly maintainer-approved merges, and direct
relevance to AI-infrastructure sponsors. Its trade-off is that GPU CI runs on Buildkite: job states are public, but
logs are not. CI-triage tasks on vLLM are therefore limited to its GitHub Actions jobs. SGLang (full CI logs) is
the natural second repository for transfer.

## Task: review concerns (`ospm/replay/concerns.py`)

- **Positive episode:** one maintainer review round with inline comments. The agent reviews the exact commit the
  maintainer reviewed, with history frozen one second before the review. It is scored on recall of the
  maintainer's (non-nit) concerns, counting a comment as matching if it is on the same file within ±5 lines, under
  a comment budget (3).
- **Negative episode:** a PR maintainers approved and merged without any maintainer inline comment. The agent should
  stay silent; this measures false alarms. AI review bots' comments are ignored.
- **AI-written text from maintainer accounts** is not ground truth (`concerns.is_ai_text`). An example is SGLang's
  `[by Codex] **Severity: style | Confidence: High**` reviews, or a pasted Codex review. A review made only of such
  comments yields no task at all. This dropped 128 concerns and 52 tasks from SGLang's newest val/test PRs, and 1
  train task from PyTorch. Backups are kept as `tasks.pre_ai_filter.jsonl`. None of the dropped tasks was in an
  earlier experiment's test set, so every SGLang result below is unchanged; rescoring confirmed this.
- **Maintainers** are reviewers with an OWNER/MEMBER/COLLABORATOR association, or anyone who approved at least two
  merged PRs. The second rule matters for vLLM, whose committers often appear as CONTRIBUTOR.
- **Splits are chronological:** the oldest 60% is train, the next 20% val, the newest 20% test.
- **No future commits.** GitHub reports some reviews, mostly approvals, against a head pushed *after* the review
  (e.g. a "merge main" days later). Such a task would show the agent code that did not exist yet, often with a
  diff of hundreds of files. Tasks whose commit postdates the event are dropped. This affected 38% of vLLM's
  negative tasks and ~5% of pytest's; the first vLLM run (v1) still had them.
- **Concern taxonomy** (`ospm/replay/concern_labels.py`, LLM-labelled at about $0.003 per 100 concerns; `--batch`
  uses the Batch API at half price).

  | | pytest (104 concerns) | vLLM (4,954) | SGLang (3,408) | PyTorch (2,570) |
  |---|---|---|---|---|
  | largest types | docs 34%, design 18%, correctness 12% | design 28%, clarification 15%, correctness 15% | correctness 24%, design 22%, clarification 13% | design 21%, correctness 18%, clarification 16%, test gaps 14% |
  | actionable | 91% | 84% | 87% | 85% |
  | need knowledge beyond the diff | 33% | 52% | **63%** | 57% |

  Concerns that need project knowledge are where learned, project-specific interventions should help. They are
  also why a diff-reading reviewer recalls so few of them on vLLM.

## Method: validated, scoped interventions (`ospm/learning/interventions.py`)

An intervention is a structured artifact, not free-text memory:

```
trigger   natural-language applicability condition
paths     glob patterns over changed files        (scope)
keywords  identifiers in the added lines           (scope)
check     what to verify
action    what to ask the author if the check fails
stage     pr_creation | review
provenance, validation {tasks, gain, fp_increase, negative_fire_rate}
```

The learning loop runs chronologically over the training PRs, in batches:

1. **Serve.** Review each PR with the active interventions whose triggers match it.
2. **Collect misses.** Gather maintainer concerns the agent missed into a rolling pool, so patterns that recur
   across time can form.
3. **Propose.** A reflector proposes scoped candidates and is told to prefer patterns from two or more PRs.
4. **Validate.** Replay *other* earlier PRs the trigger applies to, with and without the candidate: up to 3 PRs
   with maintainer concerns (from distinct PRs) and up to 3 clean ones.
   - **Pool:** the PRs learned on plus the ~150 most recent earlier ones, counted only if their outcome is already
     known at that point (a clean PR only once it has merged).
   - **Source PRs are excluded:** every review round of the PR(s) the candidate was induced from. The proposer's
     source ids are resolved to real task ids; it tends to drop prefixes, which in v1 silently disabled the
     exclusion.
   - **Adopt** only if recall on the held-out positives rises and clean PRs get no extra comments. A candidate
     that merely made the agent quieter on a clean PR is noise, not a lesson.
   - **Too broad:** triggers that fire on more than half of past clean PRs are rejected.
5. **Serve on later PRs.** Only adopted interventions whose triggers match a PR are shown to the reviewer.

**Evaluation is frozen-future.** Learn on train, then evaluate unchanged on the later test split. During learning,
a PR's lesson only becomes usable after that PR has been processed (online order).

**Baselines on identical tasks:**
- `none`: the fixed review agent, a mini-swe-agent with git and history tools.
- `memory`: ACE-style raw lesson memory. Every training concern is stored verbatim and the most similar are
  retrieved, with no structure, scope or validation.

Planned additions: GEPA on the reviewer prompt (Team-GEPA with `--entry review`), and ablations (no validation,
validation on source PRs, no scope).

## Method 2: skills evolved by post-mortem reflection (`ospm/learning/postmortem.py`)

The intervention results (below) showed that code *areas* recur while specific concerns do not, and that the
reviewer finds the right file far more often than the right lines. It also rarely consults project history. Skills
target that: a skill is a procedure for reviewing an area, not a fact about one PR.

- **Skill** (`ospm/learning/skills.py`): name, description (when to use it), body (the procedure), path globs
  (where it is offered). The agent sees an index of the skills offered for the PR's files and loads one with the
  sandbox command `skill <name>`; loads are read back from the trace.
- **`hist reviews --path <file-or-dir/>`**: earlier maintainer inline comments on those files, as of the review
  time. Procedures can tell the agent to look up what reviewers of an area usually ask.
- **Post-mortem** (GEPA-style reflective mutation, with skills as the modules). After each batch, a reflector reads,
  per PR, the maintainer concerns the agent missed (or its comments on a clean PR) next to a digest of its trace:
  commands, whether it opened the commented files, skills offered and loaded, steps used. It proposes edits, blamed
  on one part: `create` (no skill for the area), `edit` the description (offered but not loaded), `edit` the body
  (loaded but did not lead to the concern), or `delete` (causes false alarms).
- **Acceptance gate:** as for interventions. Replay other earlier PRs where the edited skill is offered (all rounds
  of the source PRs excluded), with the old vs. the new library: up to 3 PRs with concerns and 3 clean ones. Adopt
  only if recall rises and clean PRs get no extra comments; a deletion must not lose either. Edits never mutate a
  skill: each version is a new object, and cached episodes are keyed by the hash of the offered skills.
- **Systems** (`--systems` in `ospm/learning/experiment.py`): `tool` documents `hist reviews` only; `skills` adds
  the evolved library. Comparing `none`, `tool` and `skills` separates the value of the tool from that of the
  evolved procedures.
- **Team mode** (`--entry orchestrator`): the task goes to the orchestrator, which spawns specialists in the
  background (`docs/AGENT_TEAM.md`). Skills then have a role: the orchestrator's index lists coordination skills
  (when to spawn whom, what a brief must contain, when to challenge a report), and every spawned specialist gets the
  index of its own role's skills. The post-mortem additionally sees the coordination digest (the orchestrator's
  briefs and messages, each specialist's loaded skills and final report) and blames a role before editing one of its
  skills; an edit is validated where that role's skill would be offered.

## Method 3: evolving the team - agent library, agent types and coordination (`ospm/learning/team_evolution.py`)

Stage B showed why a fixed team cannot beat its own reviewer: coordination neither lost nor added findings (all 4
concerns a specialist found reached the final answer; none came from elsewhere), the security specialist never
commented, the orchestrator re-read the diff itself (median 14.5 commands) and rarely followed up, and pooling four
independent runs of the same reviewer reached only 8 of 48 concerns. The post-mortem could not learn the waste
because its objective ignored cost and its unit (path-scoped skills) could not express team-wide policy.

- **Library of agent types** (`ospm/team/roles.py`): each specialist is a type with a description (what the
  orchestrator reads to choose it), a procedure, rules and an output contract. The seed library adds `correctness`,
  `tests`, `design_docs` and a `history` researcher (contract `notes`) to the built-in reviewer and security types.
  The orchestrator's prompt lists the current library, so library edits change whom it can spawn.
- **Three levels of edits**, one per edit: *agent* (a type's rules or description), *library* (add or remove a
  type) and *team* (the orchestrator's coordination rules, and `set_board` to switch the shared notepads on or off).
- **Per-agent credit** from the traces, judged semantically: per type, how often it was spawned, its cost and
  steps, the maintainer concerns it found (and found alone), and its comments on clean PRs; for the orchestrator,
  its own cost, the episode's derived latency, bash steps and coordination calls. The reflector also sees the
  team's mean cost and latency over recent PRs.
- **Cost- and latency-aware acceptance:** replay other earlier PRs (3 with concerns from distinct PRs + 3 clean;
  source PRs excluded) with the current and the edited team. An edit is adopted in three cases:
  - quality (semantic score; clean PRs score on silence) rises by ≥ 0.01 at ≤ 1.3× the cost;
  - quality stays level while the cost falls by ≥ 15%;
  - quality stays level while the derived latency falls by ≥ 15% at ≤ 1.3× the cost.

  No edit may make episodes more than 1.5× slower. Latency is the critical path through the episode's agents
  (`ospm/harness/latency.py`, see `docs/AGENT_TEAM.md`). It uses measured tool times only when every compared
  episode has them; otherwise it uses nominal ones, so cached older episodes are compared like for like.
- **Test:** the seed library team vs. the evolved team on the same test PRs, with per-type credit on the test set.

## Measurement: semantic concern matching (`ospm/replay/semantic.py`)

The ±5-line rule counts an agent comment as matching a maintainer concern if it is on the same file within 5 lines.
It was checked against 104 hand-labelled same-file pairs (vLLM and SGLang outputs; 23 same-concern; two independent
Sonnet raters agreed with each other at κ = 0.85 and with the adjudicated labels at 97–98%):

| judge | AUC | accuracy | precision | recall | Brier | cost / 107 pairs |
|---|---:|---:|---:|---:|---:|---:|
| ±5-line rule | 0.78 | 0.83 | 0.59 | 0.70 | 0.173 | — |
| Jev (TypeSafe, `noul`) | 0.999 | 0.96 (0.99 at p ≥ 0.8) | 0.85 | 1.00 | 0.033 | $0.003 |
| OpenAI Decisions API, gpt-6-luna (`predicate`) | 1.000 | 0.97 (1.00 at p ≥ 0.1) | 1.00 | 0.87 | 0.014 | $0.005 |

Both judges are near perfect; the line rule is wrong on 41% of its hits and misses 30% of real matches. Jev is
used (threshold 0.8): it has its own rate limit (the agent runs already saturate OpenAI's tokens-per-minute) and
graded probabilities, which give a soft recall (mean best match probability per concern). Only same-file pairs are
judged, as validated. `python -m ospm.learning.rescore` re-scores a finished run from its cached outputs.

| run (positives) | system | ±5-line recall | semantic recall | soft recall | soft vs none (W/L/T) |
|---|---|---:|---:|---:|---|
| vLLM v3 (24) | none | 0.08 | 0.02 | 0.06 | — |
| | memory | 0.08 | 0.04 | 0.09 | 4/4/16 |
| vLLM 50-step (16) | none | 0.125 | 0.00 | 0.06 | — |
| | memory | — | 0.00 | 0.13 | 6/1/9 |
| SGLang (24) | none | 0.12 | 0.10 | 0.17 | — |
| | memory | 0.09 | 0.09 | 0.12 | 4/8/12 |
| | tool | 0.10 | 0.13 | 0.16 | 6/5/13 |
| | skills | 0.10 | 0.13 | 0.16 | 5/6/13 |

(Interventions equal `none` wherever nothing was adopted.) The conclusions hold under the better metric, but the
baseline is weaker than the line rule suggested: on vLLM the reviewer recovers essentially none of the maintainers'
concerns (the 50-step run's two ±5-line "hits" were a formatting nit and a mypy question matched to unrelated
comments). From here on, validation and evaluation use semantic matching.

## Which repositories can learning help? Recurrence of maintainer concerns (`ospm/replay/recurrence.py`)

Lessons, skills and memory can only help when maintainers raise the same reusable rule again on later PRs. This is
measured directly. For 200 sampled non-nit maintainer concerns of each repository's test split (its newest PRs),
the 8 most similar earlier concerns from *other* PRs are retrieved by TF-IDF (text and path). Jev then judges two
things: does an earlier comment apply the same reusable rule ("recurs"), and is that rule a convention of the
project rather than general advice the model already knows ("project-specific")?

- Two thresholds are reported, strict (P ≥ 0.8, the matching threshold) and loose (P ≥ 0.5). Terse restatements
  such as Django's "(Chop.)" for "remove the docstrings" fall in between.
- AI review text posted from maintainer accounts is excluded. On SGLang this is 15% of the newest concerns
  (`[by Codex] **Severity: style | Confidence: High**`); none were in the test sets of the SGLang experiments above.
- To compare repositories of different volume, each concern searches a random 1,000 of the earlier concerns.
- Retrieval is lexical, so the shares are lower bounds.
- Cost: under $0.40 of Jev for all seven repositories.

| repository (mirror) | recurs (strict / loose) | project-specific (strict / loose) | median gap to the earlier comment |
|---|---:|---:|---:|
| **Home Assistant** (4 months) | **26.5% / 56.5%** | **13.0% / 24.5%** | 58 days |
| SGLang (4 months) | 12.5% / 40.0% | 4.5% / 15.5% | 73 days |
| vLLM (4 months) | 10.0% / 36.0% | 1.5% / 10.5% | 64 days |
| Django (2 years) | 10.0% / 34.0% | 6.0% / 12.0% | 224 days |
| PyTorch (2 months) | 7.0% / 27.0% | 3.0% / 11.0% | 21 days |
| Rust (4 months) | 6.0% / 27.5% | 2.5% / 8.5% | 68 days |
| pytest (28 concerns only) | 3.6% / 17.9% | 0% / 0% | — |

- **Almost all recurrences are a rule applied to a different PR author** (for example 26 of the 26.5 points on Home
  Assistant), not one author's follow-up PR.
- **Home Assistant is the outlier:** two to four times the recurrence of every other repository. Its recurring
  rules are written conventions that are applied across integrations:
  - sentence case and common translation keys in `strings.json`;
  - omit the translation key when the name equals the device class;
  - tests must set up the integration and not build a coordinator directly;
  - reconfigure and reauth flows are not quality-scale exemptions;
  - do not put exception text in translatable errors.
- **Formal process alone is not enough.** Django and Rust recur no more than vLLM and SGLang, although an earlier
  hand-labelled survey ranked them high (`data/results/repo_survey/`, 13 repositories by Haiku agents with one shared
  rubric). That survey clustered generic rules ("add a test", "keep the docs accurate"), which the pairwise judge
  does not count as the same rule. Django has the highest project-specific share outside Home Assistant (release-note
  placement, docs admonitions and links).
- **A longer mirror helps, with diminishing returns.** Strict recurrence against the number of earlier concerns
  searched (project-specific in brackets):

  | earlier concerns | 250 | 500 | 1,000 | 2,000 | all (4 months) |
  |---|---:|---:|---:|---:|---:|
  | vLLM | 6.0% (0%) | 6.0% (0.5%) | 10.0% (1.5%) | 13.5% (3.0%) | 15.0% (3.0%) |
  | SGLang | 4.5% (2.0%) | 9.5% (3.5%) | 12.5% (4.5%) | 13.5% (5.0%) | 15.5% (6.5%) |
  | Home Assistant | | | 26.5% (13.0%) | | 31.5% (14.5%) |

  Recurrence about doubles from 500 to 2,000 earlier concerns, then levels off. Two to three years of vLLM would add
  a few points, not an order of magnitude.
- **The agent-tool repositories cannot be replayed for review:**
  - `openai/codex` has been an import-only mirror of an internal repository since mid-2026 and accepts no external
    PRs;
  - opencode: 1.7% of merged PRs have any human review, and bots make 85% of review comments;
  - pi: the maintainer self-merges 56% of PRs, 95% of them with no other review.

**Implication:** on vLLM, SGLang, PyTorch, pytest, Django and Rust, at most ~5% of maintainer concerns restate a
project rule that a lesson could teach. This bounds what knowledge-shaped learning could have added, and is
consistent with every result above. Home Assistant (13% strict, 25% loose) is the one repository where learned
conventions have a real chance.

## Results so far

**Significance.** Paired results now carry an exact McNemar p-value (a two-sided binomial test of wins against
losses), Benjamini–Hochberg-corrected across the systems in a run (`ospm/learning/stats.py`, as in Fan et al.
2026). Recomputed for every earlier run below, no comparison comes close to significance, with one exception:
the evolved Home Assistant rubric on the 96 test PRs with maintainer concerns (9 wins, 0 losses, p = 0.004; see
the last results).
- Home Assistant rubric (first 96 test PRs): p = 0.73.
- SGLang `tool` and `skills`: p ≥ 0.77.
- PyTorch: p ≥ 0.38.
- SGLang team evolution vs the seed team: p = 0.29.

- **pytest pipeline check:** 12 val tasks with the fixed agent scored 0.625 at $0.007 per episode, with 2 of 6
  positives matched and negatives mostly silent.
- **pytest smoke run** (16 train / 8 test). The method ran end to end and exposed the source-PR validation flaw,
  since fixed. On 8 test PRs, memory hurt (positive recall 0.30 → 0.05). Interventions tied the baseline because
  the one adopted, over-fitted intervention never fired. The sample is too small to conclude anything.
- **pytest experiment** (32 train / 24 test, chronological; $0.55):

  | system | mean score | positive recall | silent on clean PRs | paired vs none (W/L/T) |
  |---|---:|---:|---:|---|
  | none | 0.488 | 0.10 | 83% | — |
  | memory | 0.469 | 0.08 | 75% | 0/2/22 |
  | interventions | 0.488 | 0.10 | 83% | 0/0/24 |

  Seven candidates were proposed and two adopted after independent validation: "JUnit XML tests changed → check
  escaping/validity coverage" (+0.50 on 2 other PRs) and "assertion rendering changed → …" (+0.25). **Neither
  triggered on any test PR**, so the system was identical to the baseline. The raw memory slightly hurt: more false
  alarms on clean PRs, no recall gain. Lesson: on a small repository, scoped lessons rarely recur within months.
  The experiment now reports **coverage** (share of test PRs where a system injected anything), to separate "no
  effect" from "never applied". vLLM's 10–30× higher volume is the real test of whether recurrence is high
  enough.
- **vLLM, runs v1 and v2: invalid, kept for the record** (`data/results/learning_vllm_v1.*`, `_v2.*`).
  - **v1:** future-commit tasks, plus the source-exclusion bug above.
  - **v2:** 31% of review episodes (112 of 367) hit the 20-step limit and were discarded with no output, scoring 0.
    Since then the agent is warned 3 steps before the limit and gets one final turn to submit; a plain-text
    answer in that turn counts. The episode cache is versioned (`EPISODE_VERSION`) so no older episodes are
    reused.
- **vLLM v3** (`data/results/learning_vllm_v3.*`; 64 train / 48 test, chronological, 20-step reviewer; $1.66, 2.7 h). All 2958 calls succeeded, and
  every test episode produced a review.

  | system | mean score | positive recall | silent on clean PRs | paired vs none (W/L/T) | coverage |
  |---|---:|---:|---:|---|---:|
  | none | 0.500 | 0.08 | 67% | — | — |
  | memory | 0.490 | 0.08 | 58% | 1/3/44 | 96% |
  | interventions | 0.500 | 0.08 | 67% | 0/0/48 | 0% |

  **17 candidates were proposed and none adopted:**
  - 6 had no independent support: no other earlier PR matched the trigger.
  - 11 matched 1–3 other earlier PRs with concerns but raised recall on none of them.
    - **Example:** "Triton store into a newly allocated output" matched three attention-backend PRs. The
      maintainers' concerns on those were about a config file, connector design and a buffer pool.
    - **On vLLM, code areas recur but specific concerns rarely do.**

  The raw memory again slightly hurt: more comments on clean PRs and no recall gain. The baseline itself is
  weak: under the same-file, ±5-line criterion it recalls 8% of maintainer concerns.

  **Caveats:**
  - Each validation is a 3+3 replay and each test episode a single sample, so small effects are below the noise.
  - 40% of v3 episodes ended at steps 19–21, so the 20-step limit still constrained the reviewer. The step limit
    is now 50.
- **vLLM, 50-step reviewer** (`data/results/learning_vllm_s50.*`; 48 train / 32 test, both subsets of v3's;
  $1.39, 2.1 h).
  - **10 candidates were proposed and none adopted:** 4 had no independent support, and 6 gave no recall gain on
    other PRs.
  - **More steps barely changed the baseline.** On the same 32 test PRs it scored 0.508 vs 0.516 for the 20-step
    reviewer (paired 1/2/29), with recall 12.5% for both.
  - **Memory:** 0.477, paired 0/1/31.
  - **Where the reviewer looks vs what it finds.** Re-scoring the cached baseline outputs (no API cost) shows it
    comments on the file of 31–50% of maintainer concerns, but within ±5 lines of only 8–12%. It finds the right
    files and misses the specific issues, consistent with 52% of vLLM concerns needing context beyond the
    diff.
- **SGLang, 50-step reviewer** (`data/results/learning_sglang.*`; 64 train / 48 test, chronological; mirror since
  2026-06-01: 14,430 PRs, 3,712 concern tasks; ~$2.7; a laptop crash after 5 training steps, then 3.8 h resumed from the episode cache).
  All 3,187 calls of the resumed run succeeded and every test episode produced a review.

  | system | mean score | positive recall | silent on clean PRs | paired vs none (W/L/T) | coverage |
  |---|---:|---:|---:|---|---:|
  | none | 0.512 | 0.12 | 62% | — | — |
  | memory | 0.502 | 0.09 | 67% | 5/4/39 | 100% |
  | interventions | 0.512 | 0.12 | 62% | 0/0/48 | 0% |

  - **30 candidates were proposed and none adopted** (the 5 pre-crash training steps were re-proposed after the
    resume, since proposals are not cached): 11 had no independent support, 18 gave no recall gain on other PRs,
    and 1 raised comments on clean PRs.
  - The proposals were again area-specific: e.g. "new Docker image variant → add it to the release and nightly
    build matrices" was proposed twice but never helped on another PR.
  - **Memory:** recall 0.12 → 0.09 and slightly quieter on clean PRs; net 5/4/39, i.e. noise.
  - **Where the reviewer looks vs what it finds:** the baseline comments on the file of 42–44% of maintainer
    concerns (12 of 24 positive PRs have at least one file hit) but within ±5 lines of 8–12%, the same gap as
    vLLM.
  - **Conclusion across both inference engines:** at this scale, lessons validated on independent earlier PRs do
    not recur; the bottleneck is the reviewer's ability to find the specific issue, not missing project
    conventions.
- **SGLang, stage A: history tool and evolved skills, single reviewer** (`data/results/learning_sglang_skills.*`;
  same 64 train / 48 test PRs as above; `none` and `memory` episodes reused from the cache; $2.59, 3.9 h).

  | system | mean score | recall (±5 lines) | recall (file) | silent on clean PRs | paired vs none (W/L/T) |
  |---|---:|---:|---:|---:|---|
  | none | 0.512 | 0.12 | 0.44 | 62% | — |
  | memory | 0.502 | 0.09 | — | 67% | 5/4/39 |
  | tool (`hist reviews` documented) | 0.502 | 0.10 | 0.40 | 67% | 7/5/36 |
  | skills (tool + evolved library) | 0.497 | 0.10 | 0.38 | 62% | 7/6/35 |

  (± 5-line recall in the table is the per-PR mean; the file column is pooled over the 48 concerns, whose pooled
  ±5-line values are 0.08 / 0.10 / 0.10.)
  - **The tool is used but does not help.** Once documented, `hist reviews` was called in all 48 test episodes
    and changed 12 of 48 outcomes, in both directions.
  - **31 skill edits were proposed and 1 adopted.** All were `create`s (the library stayed nearly empty, so there
    was little to edit). 6 had no other past PR in their area, 21 gave no recall gain on other PRs, 3 added
    comments on clean PRs. When offered, skills *were* loaded (in most validation replays), so the gap is not
    discovery: the procedures did not lead to the concerns maintainers raised.
  - The adopted skill, `srt-environment-flags` (+0.11 recall on 3 independent PRs, no extra comments on 3 clean
    ones), applied to 5 of 48 test PRs and was loaded on 4; the skills system is within noise of `none`.
  - **Conclusion:** neither more history (the tool) nor evolved area procedures (skills) moved recall on SGLang.
    Together with the intervention results this points at the reviewer's ability to find specific issues, not at
    missing knowledge it could be told or could look up.
- **SGLang, stage B: the team** (`--entry orchestrator`; `data/results/learning_sglang_team*`; same 64 / 48 PRs).
  The orchestrator spawns specialists in the background and combines their final messages (`docs/AGENT_TEAM.md`).
  - **Training** ($5.36 over two attempts, both ended by OpenAI tokens-per-minute limits; since fixed with
    episode-level retries): 29 skill edits proposed over 8 steps, none adopted. Although the post-mortem saw the
    coordination digest, it blamed a coordination role only twice (27 reviewer skills, 1 orchestrator, 1 security):
    15 gave no gain on other PRs, 9 added comments on clean PRs, 5 had no other PR in their area. (Validation then
    still used the ±5-line rule.)
  - **Evaluation** (semantic matching, $2.03): with no skill adopted, the skills arm equals the team plus the tool.

  | system (48 test PRs) | semantic recall | soft recall | silent on clean PRs | cost / episode |
  |---|---:|---:|---:|---:|
  | single reviewer (`none`, stage A) | 0.10 | 0.17 | 62% | ~$0.010 |
  | single reviewer + tool | 0.13 | 0.16 | 67% | ~$0.010 |
  | team | 0.08 | 0.11 | 67% | ~$0.021 |
  | team + tool | 0.12 | 0.16 | 62% | ~$0.021 |

  Team + tool vs team: 6/6/36. The team matches a single reviewer at about twice the cost (as on pytest), and the
  tool again changes outcomes in both directions with no net gain.
- **SGLang, stage C: team evolution** (`data/results/team_evolution_sglang*`; 32 train / 32 test PRs, chronological,
  starting from the seed library; $6.60 training + $1.18 test, the test phase resumed from the episode cache after an
  OpenAI tokens-per-minute stop).
  - **12 edits proposed over 4 steps, 3 adopted, all for quality** on independent replays:

    | adopted edit | validation quality | cost |
    |---|---|---:|
    | rule: *"Do not spawn security by default; spawn it when the diff changes a security-sensitive boundary"* | 0.41 → 0.47 | 0.90× |
    | remove the `tests` type | 0.46 → 0.50 | 0.87× |
    | remove the `history` type | 0.50 → 0.54 | 0.85× |

    Of the 9 rejected, 4 were earlier attempts at the same removals (one, removing `tests` in step 1, lowered quality
    0.71 → 0.63); the others were a `design_docs` rule, two `design_docs` descriptions, a `security` rule and an
    orchestrator rule for CI workflow changes, none of which raised quality.
  - **Test** (32 PRs, 16 with concerns):

    | team | mean score | semantic recall | soft recall | silent on clean PRs | cost / episode |
    |---|---:|---:|---:|---:|---:|
    | seed library | 0.495 | 0.13 | 0.16 | 50% | $0.047 |
    | evolved | 0.477 | 0.13 | 0.15 | 38% | $0.035 |

    Paired 2/6/24. **The evolution removed exactly the waste the credit showed** and cut cost by 24% at the same
    recall: on the test set the seed's `security` (18 spawns), `history` (23) and `tests` (30) found no concern
    alone, and the evolved orchestrator spawned `security` on none of the 32 PRs. The loss is noise on clean PRs:
    every paired difference on a PR with concerns is a tie but one (0.25 → 0.19), and on clean PRs the evolved team
    made one extra comment on 5 and one fewer on 2. Its `design_docs` agent commented on clean PRs more often (11 vs
    6), so the next edit the credit points at is a `design_docs` rule; the training's two attempts at one were
    rejected.
  - **Conclusion:** team-level evolution works as a cost optimiser (a smaller, cheaper team at equal recall) but,
    like lessons and skills, has not raised recall: no type in the library finds what the maintainers raise.
- **PyTorch: memory, history tool and evolved skills, single reviewer** (`data/results/learning_pytorch*`; 48 train /
  48 test PRs, chronological, 24 with concerns each; semantic validation and scoring; $1.58, 6.2 h). All 2,967 calls
  succeeded and every test episode produced a review.

  | system | mean score | semantic recall | soft recall | silent on clean PRs | paired vs none (W/L/T) | coverage |
  |---|---:|---:|---:|---:|---|---:|
  | none | 0.476 | 0.01 | 0.08 | 75% | — | — |
  | memory | 0.458 | 0.00 | 0.06 | 67% | 1/4/43 | 94% |
  | tool (`hist reviews` documented) | 0.464 | 0.00 | 0.01 | 75% | 1/3/44 | 100% |
  | skills (tool + evolved library) | 0.464 | 0.00 | 0.01 | 75% | 1/3/44 | 2% |

  - **PyTorch is the hardest repository so far.** The plain reviewer matched part of one maintainer concern in 24
    PRs and wrote 0.8 comments per PR with concerns.
  - **20 skill edits proposed, 1 adopted:** 11 had no other past PR in their area, 5 gave no gain on other PRs and
    3 added comments on clean PRs. The adopted `inductor-optimization-changes` (+0.11 on 3 independent PRs, no
    extra comments on 3 clean ones) applied to 1 of 48 test PRs, where it was loaded without changing the outcome,
    so the skills arm equals the tool arm.
  - Memory and the tool changed 4–5 outcomes each, almost all on clean PRs, and lost the one partial match.
  - **Conclusion:** the same as on vLLM and SGLang. At this scale neither past reviews nor evolved procedures help
    the reviewer find the specific issues maintainers raise.
- **Home Assistant: memory, history tool and evolved skills, single reviewer** (`data/results/learning_ha_skills*`;
  64 train / 96 test PRs, chronological, 48 with concerns in the test set; mirror of 4 months, default branch `dev`;
  $1.91, 8.3 h). Chosen as the repository with the most recurring project conventions (13% strict, 25% loose,
  above). All 4,153 calls succeeded and every test episode produced a review.

  | system | mean score | semantic recall | soft recall | silent on clean PRs | paired vs none (W/L/T) | coverage |
  |---|---:|---:|---:|---:|---|---:|
  | none | 0.490 | 0.00 | 0.03 | 92% | — | — |
  | memory | 0.490 | 0.00 | 0.03 | 92% | 2/2/92 | 100% |
  | tool (`hist reviews` documented) | 0.500 | 0.00 | 0.03 | 92% | 2/1/93 | 100% |
  | skills (tool + evolved library) | 0.492 | 0.00 | 0.03 | 94% | 2/1/93 | 29% |

  - **The reviewer almost never objects.** On the 48 PRs with maintainer concerns it wrote 0.27 comments per PR
    (0.19–0.35 for the other systems) and approved most of them. Its few comments were mostly sound but different
    points, e.g. "clarify that a zero `timedelta` skips the transition" where the maintainer asked to pass `None`
    on the same line.
  - **28 skill edits proposed, 1 adopted.** Most were integration-specific (SolarEdge Modbus, a map-tile proxy,
    Google Assistant exposure migrations) and had no other past PR in their area or no gain on the PRs they had.
    The adopted `ha-entity-lifecycle` (unique-ID stability, device grouping, availability, entity metadata) is
    generic integration-review advice. It was offered on 28 test PRs and loaded on 16, without a match.
  - **The post-mortems never proposed the conventions that recur** (`strings.json` sentence case, common
    translation keys, tests that set up the integration instead of a coordinator): a post-mortem sees one missed
    concern on one PR, and validation needs that concern's rule to recur in the next few dozen PRs.
  - **Environment fix found by this run.** `grep -R` over Home Assistant's 28k files took 100–400 s on Windows (past
    the 180 s command timeout; 14% of episodes lost a search). The sandbox now runs recursive greps through
    `git grep` (`ospm/harness/fastgrep.py`; 1–3 s). Earlier repositories were barely affected (recursive-grep
    timeouts in 2.3% of SGLang episodes, about 1% on vLLM and PyTorch, none on pytest). The run above used the fix.
  - **Conclusion:** even on the repository with the most recurring conventions, the plain reviewer's recall is ~0
    and nothing the post-mortems learned changes it. The limit is mostly the reviewer's willingness to raise a
    maintainer's kind of objection; the post-mortems also never wrote the rules that recur (next result).

- **Home Assistant: a review rubric mined from the review history** (`ospm/learning/rubric.py`,
  `data/results/rubric_ha*`, `data/results/learning_ha_rubric*`; same 96 test PRs; $0.01 to write the rubric,
  $0.77 for the run, 2.8 h). This method does not wait for a post-mortem to find a rule. It mines the rules directly:
  - The recurrence judges link earlier maintainer concerns from different PRs that apply the same
    project-specific rule (1,500 sampled concerns from before the test period, 8 TF-IDF candidates each).
  - Linked concerns are clustered, and a cluster on more than 40 PRs is re-split on its stronger links.
  - One LLM call writes the 84 clusters seen on 3 or more PRs as checkable items: *rule*, *applies when*, *check*,
    and a maintainer's example.
  - The result is 58 items in 6 areas with file globs (config flows; integration tests; translations and strings;
    entities and device model; runtime architecture; manifests and quality scale). Each area is one skill,
    `rubric-<area>`.
  - Two ways to serve it: `skills` (the index of area skills; the agent loads them with `skill <name>`) or
    `rubric` (the matching areas' items inlined into the reviewer's context).

  | system | mean score | semantic recall | soft recall | comments per PR with concerns | silent on clean PRs | paired vs none (W/L/T) |
  |---|---:|---:|---:|---:|---:|---|
  | none | 0.490 | 0.00 | 0.03 | 0.27 | 92% | — |
  | skills (rubric areas on demand) | 0.503 | 0.02 | 0.05 | 0.35 | 94% | 4/2/90 |
  | **rubric (inlined)** | **0.511** | **0.05** | **0.09** | 0.50 | 90% | 5/3/88 |

  - **First system to beat the plain reviewer on Home Assistant, but within noise.**
    - Inlined, the rubric matched maintainer concerns on 3 of 48 PRs. The plain reviewer matched none.
    - Two of the three matches restate a rubric item:
      - *"Please move the `python-ecobee-api` version bump into a separate PR"*, from *submit dependency bumps
        separately from feature work*;
      - *"keep third-party exception details out of the translated `UpdateFailed` message"*, from *keep exception
        details out of translated messages*.
    - The third, *pass `user_input={}` explicitly* in a config-flow test, was also found with the skills index.
    - With 5 wins and 3 losses (sign test p ≈ 0.7), this is not yet evidence.
  - **Inlining beats loading on demand.** The area skills were offered on 90 of 96 test PRs, and the agent loaded
    at least one on 87. It still matched less with the index than with the inlined rubric. A rubric is a checklist,
    so it works when it is in front of the reviewer while they read the diff.
  - **The cost is a little more noise.** The rubric reviewer raised comments on 5 of 48 clean PRs, against 4 for
    `none`. One of these, *remove `etherscan` from the analytics fixture along with the integration*, is
    plausible but was not asked for. The reviewer is still mostly silent: 0.50 comments per PR with concerns,
    against 1.6 maintainer concerns per such PR.
  - **Mining is cheap and leak-free.** The rubric uses only concerns from before the first test PR. Writing it took
    one LLM call, and the recurrence judges' links are cached Jev judgements. Unlike post-mortem skills, the
    items are cross-integration conventions, the ones maintainers actually repeat.

- **Rubrics on more PRs and other repositories, evolved by post-mortems** (`data/results/learning_*_rubric*.json`).
  - The test set was grown to 192 Home Assistant PRs (96 with concerns, 96 clean).
  - Rubrics were mined the same way for Django (24 items, 4 areas), SGLang (30 items, 5 areas) and PyTorch
    (16 items, 6 areas).
  - The `evolved_rubric` system starts from the mined rubric. Post-mortems of the 64 training PRs then add, edit
    or remove one item at a time (`RubricLearner`).
  - Each p-value is split by PR type. On PRs with concerns a rubric can only add recall; on clean PRs it can only
    add false alarms. Mixing the two hides an effect behind the clean-PR noise.

  | repository (test PRs) | system | mean score | semantic recall | silent on clean PRs | W/L all (p) | W/L on PRs with concerns (p) |
  |---|---|---:|---:|---:|---|---|
  | Home Assistant (192) | none | 0.490 | 0.026 | 82% | — | — |
  | | mined rubric | 0.503 | 0.061 | 81% | 12/11 (1.0) | 4/0 (0.13) |
  | | **evolved rubric** | **0.517** | **0.090** | 80% | 15/10 (0.42) | **9/0 (0.004; BH 0.008)** |
  | Django (96) | none | 0.521 | 0.085 | 85% | — | — |
  | | mined rubric (= evolved) | 0.504 | 0.044 | 88% | 5/5 (1.0) | 2/3 (1.0) |
  | SGLang (48) | none | 0.502 | 0.097 | 62% | — | — |
  | | mined rubric | 0.458 | 0.042 | 58% | 3/8 (0.23) | 1/3 (0.63) |

  - **Home Assistant: the evolved rubric is the first learned system with a significant effect.**
    - It won on 9 PRs with maintainer concerns and lost on none (p = 0.004).
    - Over all 192 PRs the result is within noise: on clean PRs it had 6 wins and 10 losses, the ordinary run-to-run
      variance of whether the reviewer stays silent.
    - Only 2 of 28 edits were adopted. One sharpened *test integration behaviour through Home Assistant, not
      internal objects*; the other added *keep diagnostics out of an initial integration PR*.
  - **Django and SGLang: no effect.** The rubric made the reviewer comment more (Django 0.98 → 1.29 comments per PR
    with concerns) but match fewer concerns.
    - Their recurring project-specific concerns are 2–6% of all concerns, against 13% on Home Assistant (see the
      recurrence measure above).
    - The post-mortems adopted none of 15 edits on Django and none of 18 on SGLang.
    - The SGLang learning run stopped at step 7, when one reviewer turn ran past the 16k output-token limit. Such
      turns are now resampled up to twice (`harness.models.TruncatedTurn`), and the none/rubric comparison was run
      separately.
  - **PyTorch was dropped.** Its baseline recall is 0.01, so a rubric run would have had nothing to measure.
  - **Why so few edits were adopted.**
    - An edit was validated on the past PRs whose concerns were most similar to it: in practice one PR in the
      150-task validation pool, run once with and once without the edit.
    - The reviewer misses about 90% of concerns either way, so that one PR scored 0 both times and the gain was 0.
      In 8 cases no such PR was found at all.
    - The post-mortems also kept re-proposing rejected rules: *no single-case `parametrize`* came up 4 times.

- **Ceiling: how much could a perfect rubric help?** (`ospm/learning/ceiling.py`, `data/results/ceiling_ha.json`;
  the 96 Home Assistant test PRs with concerns; $1.02).
  - Each maintainer concern was rewritten by the LLM as the general rule it applies, without file, line or
    PR-specific detail. The reviewer was then run with the mined rubric plus those rules in two ways:
    - as one more rubric area (`oracle_items`);
    - as "maintainers found that this PR violates these rules" (`oracle_violations`).
  - The rewritten rules stay close to the concern they come from, so this is an upper bound, not a realistic rubric.

  | system | semantic recall | comments per PR | W/L vs none (p) |
  |---|---:|---:|---|
  | none | 0.026 | 0.39 | — |
  | mined rubric | 0.061 | 0.57 | 4/0 (0.13) |
  | evolved rubric | 0.090 | 0.49 | 9/0 (0.004) |
  | oracle rules as rubric items | 0.298 | 0.98 | 35/0 (<0.001) |
  | told the PR violates them | 0.670 | 1.65 | 69/0 (<0.001) |

  - **The reviewer is not the bottleneck.** Given the right items it finds 11× more concerns, and told what is
    wrong it finds two thirds of them.
  - **The gap is coverage.** The learned rubric reaches 0.09 against 0.30 for one with the right items, so a
    learner that writes and keeps more of the right items has real headroom.
  - This motivated a stronger validation gate for rubric edits:
    - **Support from all of history.** A Jev judge (`rubric.ItemJudge`) finds earlier PRs whose maintainer concerns
      apply the item, searching all of history rather than the 150-task pool. An edit with fewer than
      `--min-support` such PRs is rejected without any replay. In the first try, restricted to the pool, every edit
      found at most one such PR.
    - **Repeated sampling.** Each validation PR runs `--samples` times with and without the edit (`sample` n > 0 is
      a separate cached episode). The edit must raise the target concerns at least `--min-hit-gain` more times over
      all runs, without more comments per clean-PR run than `--max-fp-increase` allows.
    - **Memory of rejections.** Rejected edits, with their numbers, are shown to later post-mortems. Near-copies
      are rejected without a replay.
    - **Epochs.** `--epochs` passes over the training PRs. Later epochs rerun them as fresh samples, so an
      unchanged rubric does not just replay the cached outcome.
    - **Budget.** `--learn-budget` ends learning early but keeps the rubric learned so far for evaluation.

- **Home Assistant with the stronger gate** (`data/results/learning_ha_rubric_epochs*`). Settings:
  - 2 epochs; 3 runs per validation PR and arm; at least 2 supporting PRs; at least 2 extra target hits.
  - Learning stopped at its $5.50 cap after 15 of 16 steps. The run took 6.6 h and cost $6.54.

  | system (192 test PRs) | semantic recall | silent on clean PRs | W/L on PRs with concerns (p) | W/L on clean PRs |
  |---|---:|---:|---|---|
  | none | 0.026 | 82% | — | — |
  | mined rubric | 0.061 | 81% | 4/0 (0.13) | 8/11 |
  | evolved, stronger gate | 0.040 | 83% | 3/1 (0.63) | 7/7 |
  | evolved, first gate (above) | 0.090 | 80% | 9/0 (0.004) | 6/10 |

  - **The gate now runs on real evidence.**
    - Of 42 edits, 7 were rejected as not recurring and 6 as repeats of rejected edits; 29 were replayed and 4 adopted.
    - The replayed edits each had 2–3 supporting PRs; 27 of the 29 had 3.
    - The adopted edits raised their target concerns in 2–6 more of 9 runs, without more comments on clean PRs:
      - *defer diagnostics from an initial integration PR*;
      - *test through Home Assistant, not internal objects*;
      - *catch specific exceptions outside config flows*;
      - *reuse established helpers*.
  - **The adopted edits did not transfer.** On the test PRs the evolved rubric matched fewer concerns than the
    mined rubric it started from (0 wins and 2 losses against it).
  - **The earlier significant result did not replicate.** That rubric had 2 of these 4 edits and scored 9/0; this
    one, nearly a superset, scored 3/1.
    - Each test PR is run once, and the validation runs show how much one reviewer run varies. The same PR and
      context raised its target concern in 0 to 3 of 3 runs, and one edit's arms went from 7 hits to 0.
    - A 9-to-3 swing in wins is within that variance, so the p = 0.004 above is best read as a lucky draw.
  - **Validation is closer to the oracle than to the test.** The supporting PRs are the ones whose concerns the
    item states, almost verbatim, so the item tells the reviewer what to look for. That is the `oracle_items`
    condition, where recall is 0.30. On a new PR the item has to generalise, and these did not.

- **Environment fixes found by these runs.**
  - **Worktree pool.** A Home Assistant checkout takes minutes on Windows, so pooled worktrees are reused. Each gets
    a brand-new `.git` and a forced checkout and clean (`ReplayRepo.pool_size`).
  - **Prefetch by object id.** Blobs missing from the partial-clone mirror are fetched by id. This replaced a
    scratch checkout of the whole tree: 4 s instead of about 150 s per new commit.
  - **Reflog leak.** The cloned workspace's HEAD reflog named the mirror's current default-branch tip, a future
    commit, so `.git/logs` is now removed. No agent trace had ever read it.
  - **Missing grep paths.** The `git grep` shim skips missing path operands and reports them the way GNU grep
    does. They no longer fall back to the slow real grep.
