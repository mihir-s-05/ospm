"""Live tests against a real LLM API. Excluded by default; run with:

    pytest -m live            # uses OPENAI_API_KEY (gpt-6-luna) or ANTHROPIC_API_KEY (claude-opus-5-5)

Each test is a handful of calls (well under $0.01 with gpt-6-luna) and every call is capped by a UsageMeter.
"""

import os

import pytest

from ospm.agents import CITriageAgent, SecurityAgent, TriageAgent
from ospm.benchmarks.pr_data import PRS
from ospm.evolution.operators import LLMReflector, Trace
from ospm.llm import UsageMeter, make_llm
from ospm.models import CIRun, Issue, PullRequest

pytestmark = pytest.mark.live

PROVIDER = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai" if os.environ.get("OPENAI_API_KEY") else None


@pytest.fixture(scope="module")
def llm():
    if PROVIDER is None:
        pytest.skip("no ANTHROPIC_API_KEY / OPENAI_API_KEY")
    return make_llm(PROVIDER, meter=UsageMeter(budget_usd=0.25))


def test_live_triage(llm, project):
    dec = TriageAgent(project=project, llm=llm).decide(Issue(
        900, "Add a shopping cart so I can sell merch", "Please add e-commerce with Stripe checkout."))
    assert dec.used_llm, dec.extras.get("llm_error")
    assert dec.labels["decision"] == "out-of-scope"


def test_live_ci_triage(llm, project):
    log = "npm ERR! code E503\nnpm ERR! 503 Service Unavailable - GET https://registry.npmjs.org/react"
    dec = CITriageAgent(project=project, llm=llm).decide(CIRun("npm ci", "failure", log, run_id="live/npm"))
    assert dec.used_llm and dec.labels["category"] == "infra_flaky" and dec.extras["retry_recommended"]


def test_live_security_review(llm, project):
    d = next(p for p in PRS if p["id"] == "pr-9")
    dec = SecurityAgent(project=project, llm=llm).decide(PullRequest(d["number"], d["title"], diff=d["diff"]))
    assert dec.used_llm and "sql_injection" in dec.labels["findings"]
    assert any(c["finding"] == "sql_injection" for c in dec.extras["comments"])


def test_live_reflection_produces_valid_genome(llm):
    g = CITriageAgent.seed_genome()
    traces = [Trace("t1", "npm ERR! 503 Service Unavailable", {"category": "infra_flaky"},
                    {"category": "test_failure"}, 0.0, [], "", "category: predicted 'test_failure' but correct is "
                                                               "'infra_flaky'")]
    child = LLMReflector(llm, CITriageAgent, effort="low").mutate(g, traces)
    assert child is not None and child.id != g.id and child.instructions
    assert all(r.is_valid() for r in child.rules)
