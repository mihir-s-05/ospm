from .base import Agent, Case, CaseBase, FieldSpec
from .ci_triage import CITriageAgent
from .diff_agents import DiffAgent, ReviewAgent, SecurityAgent
from .planning import BacklogPlanner, ReleasePlan, plan_release
from .triage import TriageAgent

AGENTS: dict[str, type[Agent]] = {
    cls.name: cls for cls in (TriageAgent, CITriageAgent, ReviewAgent, SecurityAgent)
}

__all__ = [
    "Agent", "Case", "CaseBase", "FieldSpec", "TriageAgent", "CITriageAgent", "DiffAgent", "ReviewAgent",
    "SecurityAgent", "BacklogPlanner", "ReleasePlan", "plan_release", "AGENTS",
]
