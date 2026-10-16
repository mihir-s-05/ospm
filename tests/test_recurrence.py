"""Recurrence of maintainer concerns (ospm.replay.recurrence) with the Jev calls stubbed out."""

from __future__ import annotations

import json

from ospm.replay import recurrence
from ospm.replay.tasks import ReplayTask


def write_tasks(tmp_path, rows):
    with open(tmp_path / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for i, (pr, day, split, path, body) in enumerate(rows):
            c = {"path": path, "line": 1, "body": body, "themes": [], "nit": False}
            t = ReplayTask(f"concern:{pr}@x:{i}", "concern", f"2026-01-{day:02d}T00:00:00Z", "sha", "e",
                           {"positive": True, "concerns": [c]}, pr=pr, split=split)
            fh.write(json.dumps(t.to_dict()) + "\n")


def test_only_earlier_concerns_of_other_prs_count_and_specific_rules_are_split_out(tmp_path, monkeypatch):
    note = "add a release note in docs/releases for this change"
    write_tasks(tmp_path, [
        (1, 1, "train", "docs/releases/6.1.txt", note),
        (2, 2, "train", "django/db/query.py", "this lookup is quadratic in the number of joins"),
        (3, 9, "test", "docs/releases/6.2.txt", "please add a release note in docs/releases"),  # recurs (PR 1)
        (3, 9, "test", "django/db/query.py", "this lookup is quadratic"),  # the same PR's own text never counts
        (4, 3, "test", "django/forms/fields.py", "unrelated widget rendering bug"),
        (5, 20, "train", "docs/releases/6.2.txt", note),  # later than every target: never a candidate
    ])
    asked = []

    def rule(self, path, earlier, later):
        asked.append((earlier["pr"], later["pr"]))
        if "release note" in earlier["body"] and "release note" in later["body"]:
            return 0.95
        return 0.6 if "quadratic" in earlier["body"] and "quadratic" in later["body"] else 0.1

    monkeypatch.setattr(recurrence.RuleJudge, "_ask", rule)
    monkeypatch.setattr(recurrence.SpecificJudge, "_ask", lambda self, p, e, l: 0.9 if "release" in e["body"] else 0.2)
    r = recurrence.measure(tmp_path, sample=10, k=8)
    assert r["targets"] == 3
    assert asked and all(e != l and e != 5 for e, l in asked)
    assert r["recurring_share"] == round(1 / 3, 3) and r["project_specific_share"] == round(1 / 3, 3)
    assert r["recurring_share_loose"] == round(2 / 3, 3)  # "quadratic" (PR 2 -> PR 3) passes only the loose cut
    assert r["examples"][0]["earlier"]["pr"] == 1 and r["lag_days"]["median"] == 8.0
