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
  type) and *team* (the orchestrator's coordination rules).
- **Per-agent credit** from the traces, judged semantically: per type, how often it was spawned, its cost and
  steps, the maintainer concerns it found (and found alone), and its comments on clean PRs; for the orchestrator,
  its own cost, bash steps and coordination calls.
- **Cost-aware acceptance:** replay other earlier PRs (3 with concerns from distinct PRs + 3 clean; source PRs
  excluded) with the current and the edited team. Adopt if quality (semantic score; clean PRs score on silence)
  rises by ≥ 0.01 at ≤ 1.3× the cost, or stays level while the cost falls by ≥ 15%.
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

## Results so far

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
