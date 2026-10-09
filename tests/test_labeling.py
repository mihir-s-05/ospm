import csv
import json

import pytest

from ospm.replay import labeling


def _mk(audit, repo, items):
    (audit / repo / "dossiers").mkdir(parents=True)
    with open(audit / repo / "judgments.jsonl", "w", encoding="utf-8") as f:
        for eid, cv in items:
            (audit / repo / "dossiers" / (eid.replace(":", "_") + ".md")).write_text(
                f"# Episode {eid} (own/{repo})\n\nchain type: **x**\n\nbody text {eid}\n",
                encoding="utf-8")
            f.write(json.dumps({
                "id": eid, "chain_valid": cv, "problem_type": "p",
                "earliest_stage_with_evidence": "review", "evidence_available_then": "partial",
                "what_evidence": "SECRETEVIDENCE", "generalizes": "recurring",
                "confidence": 0.5, "notes": "SECRETNOTE"}) + "\n")


@pytest.fixture
def audit(tmp_path):
    a = tmp_path / "audit"
    _mk(a, "alpha", [("regression:1", "yes"), ("regression:2", "yes"), ("followup:3", "no")])
    _mk(a, "beta", [("regression:4", "yes"), ("followup:5", "unclear"), ("followup:6", "no")])
    return a


def test_make_kit_blind(audit, tmp_path):
    out = tmp_path / "kit"
    assert labeling.main(["make-kit", "--audit-dir", str(audit), "--out", str(out),
                          "--n", "4", "--seed", "0"]) == 0
    key = json.loads((out / "key.json").read_text())
    assert sorted(key) == ["E01", "E02", "E03", "E04"]
    assert {v["repo"] for v in key.values()} == {"alpha", "beta"}
    judg = labeling.load_judgments(audit)
    assert sum(judg[(v["repo"], v["id"])]["chain_valid"] == "yes" for v in key.values()) == 2
    for name, m in key.items():
        text = (out / "episodes" / f"{name}.md").read_text()
        assert text.startswith(f"# Labeling episode {name}")
        assert f"body text {m['id']}" in text
        assert "# Episode" not in text and m["repo"] not in text
        for bad in ("SECRET", "chain_valid", "unclear"):
            assert bad not in text
    with open(out / "rater_sheet.csv", newline="") as f:
        assert next(csv.reader(f)) == labeling.COLUMNS
    assert "key.json" in (out / "README.md").read_text()
    out2 = tmp_path / "kit2"
    labeling.make_kit(audit, out2, 4, 0)
    assert json.loads((out2 / "key.json").read_text()) == key


def test_kappa_known():
    # a: y y n n ; b: y y n y -> observed agreement po = 3/4
    # pa(y)=2/4, pb(y)=3/4 -> pe = .5*.75 + .5*.25 = .5
    # kappa = (.75 - .5) / (1 - .5) = 0.5
    assert labeling.cohen_kappa(list("yynn"), list("yyny")) == pytest.approx(0.5)
    assert labeling.cohen_kappa(list("yy"), list("yy")) == 1.0  # pe == 1, po == 1
    assert labeling.cohen_kappa(list("yy"), list("nn")) == 0.0  # pe == 0
    assert labeling.cohen_kappa(list("y"), list("y")) == 1.0


def test_agreement(audit, tmp_path):
    out = tmp_path / "kit"
    labeling.make_kit(audit, out, 4, 0)
    judg = labeling.load_judgments(audit)
    key = json.loads((out / "key.json").read_text())

    def write(name, flip):
        with open(out / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(labeling.COLUMNS)
            for i, (e, m) in enumerate(key.items()):
                cv = judg[(m["repo"], m["id"])]["chain_valid"]
                if i in flip:
                    cv = "no" if cv == "yes" else "yes"
                w.writerow([e, cv, "review", "" if i == 0 else "partial", "recurring", "", ""])

    write("rater_A.csv", set())
    write("rater_B.csv", {0})
    res = labeling.agreement(out, [out / "rater_A.csv", out / "rater_B.csv"], audit)
    ab = res["rater_A_vs_rater_B"]
    assert ab["chain_valid"]["n"] == 4 and ab["chain_valid"]["percent_agreement"] == 0.75
    assert ab["evidence_available_then"]["n"] == 3  # empty cell ignored
    assert ab["earliest_stage_with_evidence"]["kappa"] == 1.0  # degenerate case
    assert res["rater_A_vs_llm"]["chain_valid"]["percent_agreement"] == 1.0
    assert (out / "agreement.json").exists()
