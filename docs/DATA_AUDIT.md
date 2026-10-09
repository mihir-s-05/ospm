# Data-feasibility audit for cross-stage intervention learning

**Question.** Does real repository history contain enough *cross-stage chains* to support learning "where to
intervene"? A chain is a later signal (regression report, revert, follow-up fix, main-branch CI break, recurring
review correction) that can be traced to an earlier decision (a merged PR). The audit also checks whether the
evidence that would have justified an earlier intervention existed at the time, and whether those episodes can be
reconstructed.

**Setup.**
- **Repositories:** pytest-dev/pytest, pypa/pip and pydantic/pydantic, over 2026-04-01 to 2026-10-07.
- **Mirroring:** `ospm mirror` (pip and pydantic as blobless clones). That is 1,656 PRs and 439 issues, ~300MB
  on disk in total.
- **Chain mining:** `python -m ospm.replay.audit --enrich`. Detectors:
  - regression reports linked by explicit references ("caused by #N") or the SZZ algorithm (blame the lines a fix
    changed, then map them to the PR that landed them);
  - reverts;
  - "follow-up to #N" fixes;
  - main-branch CI breaks, with jobs that failed elsewhere at the same time discounted as environmental;
  - maintainer review-correction themes.

  Enrichment fetches reviews and comments of linked causing PRs from before the window.
- **Adjudication:** 52 stratified episodes, each judged by a Sonnet subagent from a dossier. A dossier holds the
  later signal, the earlier PR's text, diff, reviews and CI status, and a timeline. The rubric asks whether the
  chain is genuine, the earliest stage with evidence, whether the evidence was available then, the candidate
  intervention, its cost, its false-positive risk, and whether it generalizes.

Full tables: [`DATA_AUDIT_RESULTS.md`](DATA_AUDIT_RESULTS.md). Raw data: `data/audit/<repo>/{summary.json,
chains.jsonl,judgments.jsonl,dossiers/}`.

## Findings

1. **Genuine chains are rare.** 18 of 52 adjudicated episodes (35%) were genuine. Extrapolating precision by
   method gives roughly 4 (pip), 13 (pydantic) and 13 (pytest) genuine chains in six months. That is about 30
   across three large, active repositories, against ~1,650 PRs opened in the window: a base rate of roughly 2%. Hard negatives are
   plentiful; positives are scarce.
2. **Automatic linking needs adjudication.** Precision depends strongly on the method:

   | method | genuine / judged |
   |---|---:|
   | follow-up references ("follow-up to #N", "broke in #N") | 11 / 17 (65%) |
   | regression report + SZZ blame | 3 / 7 (43%) |
   | regression report, explicit reference | 1 / 1 |
   | regression report with no attributable PR | 3 / 12 |
   | main-branch CI break right after a merge | **0 / 14** |
   | revert | 0 / 1 (1 unclear) |

   Main-branch breaks were environmental (pre-release Pythons, platform runners, dependency builds) or followed a
   docs-only PR. This held even after filtering jobs that failed elsewhere. They are not a usable learning signal.
3. **The evidence is almost always at PR review, never at issue refinement.** Of the 18 genuine chains, the
   earliest stage with evidence was review in 14, PR creation in 2, pre-merge CI in 1, and nowhere in 1.
   **None** could have been caught at issue refinement. Evidence was fully available in 9 cases and partly in 8.
4. **The candidate interventions are cheap, recurring and mostly precise.** 14 of 18 were judged recurring, and
   most carried low or medium false-positive risk. Typical ones:
   - boundary-value, round-trip or parity tests when logic is replaced (serializer output vs. schema, old vs. new
     dispatch, literal vs. computed character ranges);
   - "check sibling options" when one option is wired in;
   - do not merge with an unaddressed CHANGES_REQUESTED review or an unanswered bot finding on a public hook;
   - fresh CI before merging a long-stale PR;
   - import-time checks for new top-level imports.

   These fit the proposed (trigger, evidence, action, stage) representation well.
5. **Feedback is long-delayed, and historical CI evidence for causing PRs is gone.** The median delay from merge
   to regression report is 408 days (pytest), 92 (pydantic) and 49 (pip). Causing PRs usually lie outside a
   six-month window. Their text and reviews can be recovered through the API, but **none** of their CI logs can
   (GitHub deletes them after ~90 days).
6. **Review corrections are the densest signal.** Maintainer review comments fall into recurring themes. In
   pytest: docs 23 PRs, tests 17, compatibility 15, typing 14, changelog 13. In pip: naming/style 14, docs 11,
   edge cases 10. These are immediate, with no long delay, and far more frequent than regressions.

## Implications for the research plan

- **Narrow the cross-stage claim to "downstream signal → PR creation/review".** In these repositories,
  issue-stage interventions have no empirical support.
- **Use two data sources with different strengths:**
  1. Recurring review corrections, which are frequent, immediate and plentiful. Use them to learn PR-creation
     checks that pre-empt reviewer comments.
  2. Regression and follow-up chains, which are rare and high-value. Use them as the hard test of whether learned
     interventions catch real defects earlier.
- **Scale data deliberately.** Extend the window to 2–3 years (PR text and reviews remain available) and add
  5–10 repositories, or a sponsor repository with archived CI. About 30 genuine chains per six months across three
  repositories is too few for chronological train/dev/test splits. Start archiving CI logs prospectively now.
- **Drop main-branch breaks as a signal; keep follow-ups and explicit references.** Treat SZZ links as candidates
  that need adjudication.
- **Validate the adjudicator.** Every label here comes from one LLM annotator. Before using these labels in a
  paper, have a human spot-check a sample (for example 20 episodes, two raters) and report agreement.

## Limitations

- Recall is unknown: regressions that never say "regression", and fixes that do not reference their cause, are
  missed.
- The adjudication sample is small (52) and single-annotator.
- pytest's sample includes 7 dossiers from an earlier detector version.
- The window is six months; the conclusions about delays and log availability are the robust part.
