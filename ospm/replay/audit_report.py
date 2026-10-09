"""Combine audit summaries, mined chains and adjudications into a feasibility report.

    python -m ospm.replay.audit_report data/audit --out docs/DATA_AUDIT_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def link_method(chain: dict) -> str:
    notes = chain.get("notes") or ""
    if chain["kind"] == "regression":
        if "SZZ" in notes:
            return "regression/SZZ"
        if "explicit" in notes:
            return "regression/explicit-reference"
        return "regression/unlinked"
    return chain["kind"]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def analyse(audit_dir: Path) -> dict:
    repos = {}
    for d in sorted(p for p in audit_dir.iterdir() if p.is_dir() and (p / "summary.json").exists()):
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        chains = {c["id"]: c for c in load_jsonl(d / "chains.jsonl")}
        judged = load_jsonl(d / "judgments.jsonl")
        by_method: dict[str, Counter] = defaultdict(Counter)
        for j in judged:
            c = chains.get(j["id"])
            method = link_method(c) if c else "unknown"
            by_method[method][j["chain_valid"]] += 1
        precision = {m: round(c["yes"] / sum(c.values()), 2) for m, c in by_method.items() if sum(c.values())}
        linked_counts = Counter(link_method(c) for c in chains.values()
                                if c.get("linked") and c["kind"] in ("regression", "revert", "followup", "main_break"))
        est_valid = {m: round(n * precision.get(m, 0.0), 1) for m, n in linked_counts.items()}
        valid = [j for j in judged if j["chain_valid"] == "yes"]
        repos[d.name] = {
            "summary": summary, "judged": len(judged),
            "chain_valid": Counter(j["chain_valid"] for j in judged),
            "by_method": {m: dict(c) for m, c in by_method.items()}, "precision_by_method": precision,
            "linked_by_method": dict(linked_counts), "estimated_valid_chains": est_valid,
            "estimated_valid_total": round(sum(est_valid.values()), 1),
            "valid_stage": Counter(j["earliest_stage_with_evidence"] for j in valid),
            "valid_evidence": Counter(j["evidence_available_then"] for j in valid),
            "valid_problem": Counter(j["problem_type"] for j in valid),
            "valid_generalizes": Counter(j["generalizes"] for j in valid),
            "valid_fp_risk": Counter(j["false_positive_risk"] for j in valid),
            "interventions": [(j["id"], j["earliest_stage_with_evidence"], j["candidate_intervention"],
                               j["generalizes"], j["false_positive_risk"]) for j in valid],
        }
    return repos


def render(repos: dict) -> str:
    L = ["# Data-feasibility audit: results", ""]
    L += ["| repo | window issues / PRs | linked chains (auto) | adjudicated | genuine | auto-link precision | "
          "est. genuine chains | review themes (#PRs) |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for name, r in repos.items():
        s = r["summary"]
        linked = sum(r["linked_by_method"].values())
        cv = r["chain_valid"]
        prec = round(cv.get("yes", 0) / r["judged"], 2) if r["judged"] else None
        themes = ", ".join(f"{k} {v}" for k, v in list(s["review_themes"]["themes"].items())[:4])
        L.append(f"| {name} | {s['window_issues']} / {s['window_prs']} | {linked} | {r['judged']} | {cv.get('yes', 0)} | "
                 f"{prec} | ~{r['estimated_valid_total']} | {themes} |")
    L += ["", "## Precision of automatic linking, by method (adjudicated sample)", "",
          "| repo | method | judged | yes | no | unclear |", "|---|---|---:|---:|---:|---:|"]
    for name, r in repos.items():
        for m, c in sorted(r["by_method"].items()):
            L.append(f"| {name} | {m} | {sum(c.values())} | {c.get('yes', 0)} | {c.get('no', 0)} | {c.get('unclear', 0)} |")
    L += ["", "## Genuine chains: where was the evidence?", ""]
    for name, r in repos.items():
        L.append(f"- **{name}**: stage {dict(r['valid_stage'])}; evidence {dict(r['valid_evidence'])}; "
                 f"problem {dict(r['valid_problem'])}; generalizes {dict(r['valid_generalizes'])}; "
                 f"false-positive risk {dict(r['valid_fp_risk'])}")
    L += ["", "## Candidate interventions from genuine chains", "",
          "| repo | episode | stage | intervention | generalizes | FP risk |", "|---|---|---|---|---|---|"]
    for name, r in repos.items():
        for eid, stage, iv, gen, fp in r["interventions"]:
            L.append(f"| {name} | {eid} | {stage} | {iv} | {gen} | {fp} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audit_dir")
    p.add_argument("--out")
    a = p.parse_args(argv)
    repos = analyse(Path(a.audit_dir))
    text = render(repos)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    (Path(a.audit_dir) / "report.json").write_text(json.dumps(repos, indent=1, default=dict), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
