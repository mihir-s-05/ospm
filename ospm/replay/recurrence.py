"""Recurrence of maintainer review concerns: how often a concern on a newer PR restates a rule a maintainer already
applied on an older, different PR.

Lessons, skills and memory can only help a reviewer when concerns recur. For each sampled concern of the test split
(the newest PRs), the `K` most similar earlier concerns from other PRs are retrieved by TF-IDF (text + file path),
and Jev judges each pair: does the earlier comment apply the same reusable rule? For pairs that do, a second question
asks whether the rule is a convention of this project rather than general good practice (which the model already
knows). Retrieval is lexical, so the shares are lower bounds.

    python -m ospm.replay.recurrence --data data/django data/rust --sample 200 --out data/results/recurrence.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from datetime import datetime
from pathlib import Path

from .concerns import AI_TEXT
from .semantic import THRESHOLD, Matcher
from .tasks import load_tasks

K = 8
LOOSE = 0.5  # Jev's 0.8 threshold misses terse restatements ("(Chop.)"), so both are reported


class RuleJudge(Matcher):
    """P(the later maintainer comment applies the same reusable rule as the earlier one), cached like Matcher."""

    QUESTION = ("Do these two maintainer review comments, made on DIFFERENT pull requests of the same project, apply "
                "the same reusable review rule or requirement, so that a reviewer who had learned the rule from the "
                "earlier comment would raise the later comment's point on the later pull request?")
    YES = ("Same rule: both ask for the same kind of change for the same reason (e.g. both require a release note in "
           "the same place, both forbid the same API or pattern, both ask for the same kind of test or check).")
    NO = ("Different: different problems or requests, or only the same broad category (two unrelated bugs, two "
          "unrelated 'add a test' requests about different behaviour, two different design objections).")

    def _state(self, path: str, earlier: dict, later: dict) -> dict:
        return {"earlier_comment": {"file": earlier.get("path"), "text": str(earlier.get("body", ""))[:1500]},
                "later_comment": {"file": later.get("path"), "text": str(later.get("body", ""))[:1500]}}


class SpecificJudge(RuleJudge):
    """P(the shared rule is a convention of this project, not general good practice)."""

    QUESTION = ("These two review comments from different pull requests apply the same rule. Is that rule a "
                "convention or requirement specific to THIS project (its own APIs, helpers, file layout, release "
                "notes or changelog process, naming, test fixtures, deprecation policy, documentation markup), as "
                "opposed to general programming advice any experienced reviewer would give on any project?")
    YES = "Project-specific: someone new to the project would have to learn it from the project or its maintainers."
    NO = "General good practice: e.g. 'add a test', 'handle the error', 'simplify this', 'fix the typo'."


def concerns_of(data: Path) -> list[dict]:
    out = []
    for t in load_tasks(data / "tasks.jsonl"):
        if t.kind != "concern" or not t.truth.get("positive"):
            continue
        author = (re.search(r" by @(\S+) is ready", t.event) or [None, ""])[1]
        for c in t.truth["concerns"]:
            if not c.get("nit") and (c.get("body") or "").strip() and not AI_TEXT.search(c["body"]):
                out.append({"pr": t.pr, "author": author, "as_of": t.as_of, "split": t.split, "path": c["path"],
                            "body": c["body"], "line": c.get("line"), "themes": c.get("themes") or []})
    return out


def _doc(c: dict) -> str:
    return c["body"] + " " + " ".join(c["path"].replace("/", " ").replace(".", " ").replace("_", " ").split())


def candidates(cs: list[dict], targets: list[int], k: int = K, seed: int = 0,
               pool: int | None = None) -> list[tuple[int, int]]:
    """(target, earlier) index pairs: each target's `k` most similar earlier concerns of other PRs (TF-IDF over
    text + path). `pool` first samples that many earlier concerns, so busier histories compete on equal footing."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    m = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english").fit_transform([_doc(c) for c in cs])
    out = []
    for i in targets:
        sims = linear_kernel(m[i], m).ravel()
        earlier = [j for j in range(len(cs)) if cs[j]["as_of"] < cs[i]["as_of"] and cs[j]["pr"] != cs[i]["pr"]]
        if pool and len(earlier) > pool:
            earlier = random.Random(f"{seed}:{i}").sample(earlier, pool)
        out += [(i, j) for j in sorted(earlier, key=lambda j: -sims[j])[:k]]
    return out


def judges(data: Path) -> tuple[RuleJudge, SpecificJudge]:
    return RuleJudge(data / "recurrence_cache.jsonl"), SpecificJudge(data / "recurrence_specific_cache.jsonl")


def measure(data: Path, sample: int = 200, k: int = K, seed: int = 0, pool: int | None = None) -> dict:
    """Share of sampled test-split concerns that restate an earlier concern's rule (and a project-specific one)."""
    cs = sorted(concerns_of(data), key=lambda c: c["as_of"])
    targets = [i for i, c in enumerate(cs) if c["split"] == "test"]
    random.Random(seed).shuffle(targets)
    targets = targets[:sample]
    rule, spec = judges(data)
    owner = candidates(cs, targets, k, seed, pool)
    p_rule = rule.probabilities([("", cs[j], cs[i]) for i, j in owner])
    best: dict[int, tuple[float, int]] = {}
    for (i, j), p in zip(owner, p_rule):
        if p > best.get(i, (-1.0, -1))[0]:
            best[i] = (p, j)
    loose = {i: j for i, (p, j) in best.items() if p >= LOOSE}
    hits = {i: j for i, j in loose.items() if best[i][0] >= THRESHOLD}
    p_spec = spec.probabilities([("", cs[j], cs[i]) for i, j in loose.items()])
    specific = {i for i, p in zip(loose, p_spec) if p >= 0.5}

    def days(i: int, j: int) -> float:
        f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
        return (f(cs[i]["as_of"]) - f(cs[j]["as_of"])).total_seconds() / 86400

    lags = [days(i, j) for i, j in hits.items()]
    n = len(targets)
    other = {i for i, j in loose.items() if cs[i]["author"] != cs[j]["author"]}  # a rule, not one author's follow-up
    return {"data": str(data), "concerns": len(cs), "targets": n, "pairs_judged": len(owner),
            "recurring_share": round(len(hits) / n, 3) if n else None,
            "project_specific_share": round(len(specific & set(hits)) / n, 3) if n else None,
            "recurring_share_loose": round(len(loose) / n, 3) if n else None,
            "project_specific_share_loose": round(len(specific) / n, 3) if n else None,
            "other_author_share": round(len(other & set(hits)) / n, 3) if n else None,
            "other_author_share_loose": round(len(other) / n, 3) if n else None,
            "pool": pool,
            "lag_days": {"median": round(statistics.median(lags), 1), "p75": round(sorted(lags)[3 * len(lags) // 4], 1),
                         "max": round(max(lags), 1)} if lags else None,
            "history_days": round(days(len(cs) - 1, 0), 1) if cs else 0,
            "judge_tokens": rule.tokens + spec.tokens,
            "examples": [{"specific": i in specific, "lag_days": round(days(i, j), 1),
                          "earlier": {"pr": cs[j]["pr"], "path": cs[j]["path"], "body": cs[j]["body"][:300]},
                          "later": {"pr": cs[i]["pr"], "path": cs[i]["path"], "body": cs[i]["body"][:300]}}
                         for i, j in list(hits.items())[:12]]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=200, help="test-split concerns per repository")
    ap.add_argument("--k", type=int, default=K, help="earlier candidates judged per concern")
    ap.add_argument("--pool", type=int, help="random sample of earlier concerns to search (equal footing)")
    ap.add_argument("--out", default="data/results/recurrence.json")
    a = ap.parse_args(argv)
    res = {}
    for d in a.data:
        r = measure(Path(d), a.sample, a.k, pool=a.pool)
        res[Path(d).name] = r
        print(f"[recurrence] {Path(d).name}: {r['targets']} concerns, recurring {r['recurring_share']}, "
              f"project-specific {r['project_specific_share']} (loose {r['recurring_share_loose']} / "
              f"{r['project_specific_share_loose']}), other author {r['other_author_share']} / "
              f"{r['other_author_share_loose']}, lag {r['lag_days']}, history {r['history_days']} d",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
