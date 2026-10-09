"""Text utilities, diff parsing, item serialisation, genomes and rules."""

import pytest

from ospm.genome import Genome, Rule, match_rules
from ospm.models import CIRun, Issue, PullRequest, item_from_dict, item_to_dict, parse_diff
from ospm.text import TfidfIndex, content_tokens, normalize, phrase_in, tokenize, truncate


# ---------------------------------------------------------------------------------------------- text
def test_tokenize_and_phrase_matching_ignore_punctuation():
    assert tokenize("subprocess.run(cmd, shell=True)") == ["subprocess", "run", "cmd", "shell", "true"]
    norm = normalize("subprocess.run(cmd, shell=True)")
    assert phrase_in("shell=True", norm)
    assert phrase_in("shell true", norm)
    assert not phrase_in("shell false", norm)
    assert not phrase_in("hell", norm)  # whole tokens only
    assert not phrase_in("", norm)


def test_content_tokens_drop_stopwords_and_numbers():
    assert content_tokens("The build is broken on 3.12 and I can't fix it") == ["build", "broken", "fix"]


def test_truncate_keeps_head_and_tail():
    s = "a" * 100 + "b" * 100
    out = truncate(s, 80)
    assert out.startswith("a") and out.endswith("b") and "[truncated]" in out
    assert truncate("short", 80) == "short"


def test_tfidf_ranks_similar_documents_and_excludes():
    idx = TfidfIndex()
    idx.add("a", "live reload breaks when editing theme partials")
    idx.add("b", "rss feed timezone is wrong in pubDate")
    idx.add("c", "partials do not trigger live reload")
    res = idx.query("live reload not triggered for partials", k=2)
    assert {r[0] for r in res} == {"a", "c"}
    assert all(r[1] > 0 for r in res)
    assert "c" not in [r[0] for r in idx.query("live reload partials", k=3, exclude={"c"})]
    assert idx.query("", k=3) == []


# ---------------------------------------------------------------------------------------------- diffs
DIFF = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -10,3 +10,4 @@ def f():
 keep = 1
--- removed sql comment
+added_one = 2
+++ added line that starts with plus signs
 tail = 3
diff --git a/tests/test_app.py b/tests/test_app.py
--- /dev/null
+++ b/tests/test_app.py
@@ -0,0 +1,2 @@
+def test_x():
+    assert True
"""


def test_parse_diff_line_numbers_and_tricky_prefixes():
    files = parse_diff(DIFF)
    assert [f.path for f in files] == ["src/app.py", "tests/test_app.py"]
    app = files[0]
    assert app.added == [(11, "added_one = 2"), (12, "++ added line that starts with plus signs")]
    assert app.removed == ["-- removed sql comment"]
    assert files[1].added == [(1, "def test_x():"), (2, "    assert True")]


def test_pull_request_files_and_item_roundtrip():
    pr = PullRequest(number=7, title="t", diff=DIFF, ci_runs=[CIRun("tests", "failure", "log", run_id="pr-7/tests")])
    assert pr.files == ["src/app.py", "tests/test_app.py"]
    back = item_from_dict(item_to_dict(pr))
    assert back == pr and isinstance(back.ci_runs[0], CIRun)
    issue = Issue(number=1, title="x", body="y", labels=["bug"])
    assert item_from_dict(item_to_dict(issue)) == issue
    with pytest.raises(TypeError):
        item_to_dict(object())


# ---------------------------------------------------------------------------------------------- rules & genomes
def test_rule_matching_semantics():
    r = Rule(id="r", field="kind", value="bug", any_of=["crash", "exception"], none_of=["question"])
    assert r.matches("app crashes", normalize("app crash")) is True
    assert r.matches("", normalize("how do I fix this exception question")) is False  # vetoed
    assert r.matches("", normalize("works fine")) is False
    both = Rule(id="b", field="kind", value="bug", all_of=["live reload", "partial"])
    assert both.matches("", normalize("live reload ignores partial edits"))
    assert not both.matches("", normalize("live reload broken"))
    pat = Rule(id="p", field="f", value="v", pattern=r"^\s*print\(")
    assert pat.matches("x = 1\n    print(x)", normalize("x"))
    assert not pat.matches("log(print(x))", normalize("x"))
    assert not Rule(id="e", field="f", value="v").matches("anything", normalize("anything"))


def test_rule_validity_and_from_dict():
    assert not Rule(id="x", field="f", value="v", pattern="(unclosed").is_valid()
    assert not Rule(id="x", field="f", value="v").is_valid()
    r = Rule.from_dict({"id": "x", "field": "f", "value": "v", "any_of": ["a", " ", ""], "weight": "2", "junk": 1})
    assert r.any_of == ["a"] and r.weight == 2.0


def test_genome_id_is_content_addressed():
    g = Genome(agent="triage", instructions="be careful", params={"knn_k": 5})
    g.add_rule(Rule(id="r1", field="kind", value="bug", any_of=["crash"]))
    h = Genome.from_json(g.to_json())
    assert h.id == g.id
    h.notes = "notes and lineage do not change behaviour"
    h.parents = ["abc"]
    assert h.id == g.id
    h.params["knn_k"] = 6
    assert h.id != g.id
    child = g.clone()
    assert child.parents == [g.id] and child.generation == g.generation + 1 and child.id == g.id


def test_add_rule_dedups_and_renames_ids():
    g = Genome(agent="triage", instructions="")
    assert g.add_rule(Rule(id="r", field="kind", value="bug", any_of=["crash"]))
    assert not g.add_rule(Rule(id="other", field="kind", value="bug", any_of=["CRASH"]))  # same signature
    assert g.add_rule(Rule(id="r", field="kind", value="bug", any_of=["panic"]))
    assert [r.id for r in g.rules] == ["r", "r-2"]
    assert not g.add_rule(Rule(id="bad", field="kind", value="bug", pattern="("))
    assert [r.id for r in match_rules(g.rules, "kernel panic")] == ["r-2"]
