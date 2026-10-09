from ospm.replay.survey import landed, recurrence


def test_landed_counts_bot_landed_prs():
    assert landed({"merged_at": "2026-01-01T00:00:00Z", "labels": []})
    assert landed({"merged_at": None, "labels": [{"name": "Merged"}]})  # e.g. PyTorch's mergebot
    assert not landed({"merged_at": None, "labels": [{"name": "triaged"}]})


def test_recurrence_ignores_matches_within_the_same_pr():
    same_pr = [(1, "please add a changelog entry for this fix"), (1, "please add a changelog entry for this fix")]
    assert recurrence(same_pr)["frac_sim_ge_0.5"] == 0.0
    across = [(1, "please add a changelog entry for this fix"), (2, "please add a changelog entry for this change"),
              (3, "why does the scheduler hold this lock while waiting on the kv transfer")]
    r = recurrence(across)
    assert r["comments"] == 3 and r["frac_sim_ge_0.5"] == round(2 / 3, 3)
