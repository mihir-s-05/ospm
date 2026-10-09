"""Blind human-labelling kit and inter-rater agreement for audit episodes.

    python -m ospm.replay.labeling make-kit --audit-dir data/audit --out data/labeling --n 20 --seed 0
    python -m ospm.replay.labeling agreement --kit data/labeling --raters a.csv b.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

COLUMNS = ["episode", "chain_valid", "earliest_stage_with_evidence",
           "evidence_available_then", "generalizes", "confidence", "notes"]
CATEGORICAL = ["chain_valid", "earliest_stage_with_evidence",
               "evidence_available_then", "generalizes"]

README = """# Human labeling kit

You are labeling {n} maintenance episodes. Each file in `episodes/` (E01.md, ...)
is one episode dossier. Fill in one row per episode in your own copy of
`rater_sheet.csv` (e.g. `rater_A.csv`, `rater_B.csv`).

**Do NOT open `key.json`.** It maps episodes to repos/ids and exists only so the
analysis can be computed later; opening it breaks blindness.

Each rater fills their own copy independently. Do not discuss episodes or
answers with the other rater until both sheets are submitted.

## Rubric (allowed values per column)

- `chain_valid`: `yes` | `no` | `unclear`
  (is the described chain of earlier change -> later problem real?)
- `earliest_stage_with_evidence`: `issue_refinement` | `pr_creation` | `review` |
  `pre_merge_ci` | `post_merge_only` | `none`
- `evidence_available_then`: `yes` | `partial` | `no`
- `generalizes`: `recurring` | `one_off`
- `confidence`: a number from 0 to 1 (or low/medium/high; be consistent)
- `notes`: free text, optional

## No-hindsight rule

"Evidence available then" means information that existed BEFORE the earlier PR
was merged. Anything learned only from the later signal, later fixes, or
discussion after the merge must not count as evidence available then.

Leave a cell empty only if you truly cannot answer; empty cells are ignored
in agreement statistics.
"""


def load_judgments(audit_dir: Path) -> dict[tuple[str, str], dict]:
    res = {}
    for repo_dir in sorted(p for p in Path(audit_dir).iterdir() if p.is_dir()):
        jp = repo_dir / "judgments.jsonl"
        if jp.exists():
            for line in jp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    j = json.loads(line)
                    res[(repo_dir.name, str(j["id"]))] = j
    return res


def _take_round_robin(pools, quota, rng, repo_order):
    pools = {r: list(v) for r, v in pools.items()}
    for v in pools.values():
        rng.shuffle(v)
    chosen = []
    while len(chosen) < quota and any(pools.values()):
        for r in repo_order:
            if len(chosen) >= quota:
                break
            if pools.get(r):
                chosen.append((r, pools[r].pop()))
    return chosen


def select_episodes(audit_dir: Path, n: int, seed: int) -> list[tuple[str, str]]:
    judgments = load_judgments(audit_dir)
    buckets: dict[str, dict[str, list[str]]] = {"yes": {}, "other": {}}
    for (repo, eid), j in sorted(judgments.items()):
        fname = eid.replace(":", "_") + ".md"
        if (Path(audit_dir) / repo / "dossiers" / fname).exists():
            b = "yes" if j.get("chain_valid") == "yes" else "other"
            buckets[b].setdefault(repo, []).append(eid)
    rng = random.Random(seed)
    repos = sorted({r for b in buckets.values() for r in b})
    rng.shuffle(repos)
    avail_yes = sum(len(v) for v in buckets["yes"].values())
    avail_other = sum(len(v) for v in buckets["other"].values())
    n_yes = min(n // 2, avail_yes)
    n_other = min(n - n_yes, avail_other)
    n_yes = min(n - n_other, avail_yes)
    rot = repos[1:] + repos[:1]  # rotate so odd quotas balance across repos
    chosen = _take_round_robin(buckets["yes"], n_yes, rng, repos)
    chosen += _take_round_robin(buckets["other"], n_other, rng, rot)
    rng.shuffle(chosen)  # episode numbers must not leak repo/validity
    return chosen


def make_kit(audit_dir: Path, out: Path, n: int, seed: int) -> dict:
    audit_dir, out = Path(audit_dir), Path(out)
    chosen = select_episodes(audit_dir, n, seed)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    key = {}
    for i, (repo, eid) in enumerate(chosen, 1):
        name = f"E{i:02d}"
        src = audit_dir / repo / "dossiers" / (eid.replace(":", "_") + ".md")
        lines = src.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith("# Episode"):
            lines = lines[1:]  # drop original header carrying id and repo
        body = "\n".join(lines).lstrip("\n")
        (out / "episodes" / f"{name}.md").write_text(
            f"# Labeling episode {name}\n\n{body}\n", encoding="utf-8")
        key[name] = {"repo": repo, "id": eid}
    (out / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    with open(out / "rater_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for name in key:
            w.writerow([name] + [""] * (len(COLUMNS) - 1))
    (out / "README.md").write_text(README.format(n=len(key)), encoding="utf-8")
    return key


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def compare(a: dict[str, str], b: dict[str, str]) -> dict:
    keys = [k for k in a if k in b and a[k] and b[k]]
    x = [a[k] for k in keys]
    y = [b[k] for k in keys]
    n = len(keys)
    return {"n": n,
            "percent_agreement": (sum(p == q for p, q in zip(x, y)) / n) if n else None,
            "kappa": cohen_kappa(x, y) if n else None}


def read_rater(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["episode"].strip(): {c: (r.get(c) or "").strip().lower()
                                       for c in CATEGORICAL}
                for r in csv.DictReader(f) if r.get("episode")}


def agreement(kit: Path, raters: list[Path], audit_dir: Path | None) -> dict:
    kit = Path(kit)
    key = json.loads((kit / "key.json").read_text(encoding="utf-8"))
    data = {Path(p).stem: read_rater(Path(p)) for p in raters}
    names = list(data)

    def col(rows, c):
        return {e: r[c] for e, r in rows.items()}

    result: dict = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            result[f"{names[i]}_vs_{names[j]}"] = {
                c: compare(col(data[names[i]], c), col(data[names[j]], c))
                for c in CATEGORICAL}
    if audit_dir is not None:
        judgments = load_judgments(Path(audit_dir))
        llm = {}
        for e, m in key.items():
            jd = judgments.get((m["repo"], m["id"]))
            if jd:
                llm[e] = {c: str(jd.get(c) or "").strip().lower() for c in CATEGORICAL}
        for name, rows in data.items():
            result[f"{name}_vs_llm"] = {c: compare(col(rows, c), col(llm, c))
                                        for c in CATEGORICAL}
    (kit / "agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def print_table(result: dict) -> None:
    print(f"{'comparison':<24}{'column':<32}{'n':>4}{'agree%':>9}{'kappa':>8}")
    for pair, cols in result.items():
        for c, s in cols.items():
            pa = "-" if s["percent_agreement"] is None else f"{100 * s['percent_agreement']:.1f}"
            k = "-" if s["kappa"] is None else f"{s['kappa']:.3f}"
            print(f"{pair:<24}{c:<32}{s['n']:>4}{pa:>9}{k:>8}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ospm.replay.labeling")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-kit")
    mk.add_argument("--audit-dir", default="data/audit")
    mk.add_argument("--out", default="data/labeling")
    mk.add_argument("--n", type=int, default=20)
    mk.add_argument("--seed", type=int, default=0)
    ag = sub.add_parser("agreement")
    ag.add_argument("--kit", default="data/labeling")
    ag.add_argument("--raters", nargs="+", required=True)
    ag.add_argument("--audit-dir", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "make-kit":
        key = make_kit(Path(args.audit_dir), Path(args.out), args.n, args.seed)
        print(f"wrote {len(key)} episodes to {args.out}")
    else:
        print_table(agreement(Path(args.kit), [Path(r) for r in args.raters],
                              Path(args.audit_dir) if args.audit_dir else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
