"""Semantic concern matching: does an agent's review comment raise the same concern as a maintainer's?

The ±5-line rule used for scoring is a poor proxy: on 104 hand-labelled same-file pairs (vLLM, SGLang) it had
precision 0.59 and recall 0.70, while Jev (TypeSafe's System One model, a calibrated true/false "noul") reached
AUC 0.999 and 99% accuracy at a 0.8 threshold. Only same-file pairs are judged, the setting it was validated on.
Answers are cached on disk (input-token priced, ~$0.03 per thousand pairs).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"
THRESHOLD = 0.8
QUESTION = ("Does the AI reviewer's comment raise the SAME concern as the maintainer's comment on this pull "
            "request (the same underlying problem or request, even if worded differently or anchored a few lines "
            "away), such that acting on the AI comment would make the author address the maintainer's point?")
YES = "Same concern: both point out the same bug, the same missing check/test/doc, or the same design objection."
NO = ("Different concern: a different issue, even if in the same function or area, or only loosely related. "
      "Line numbers alone do not decide it.")


def _key(path: str, concern: dict, comment: dict) -> str:
    blob = json.dumps([path, concern.get("line"), concern.get("body"), comment.get("line"), comment.get("body")])
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


class Matcher:
    """P(same concern) for (maintainer concern, agent comment) pairs on the same file, cached in `cache_path`."""

    def __init__(self, cache_path: str | Path, workers: int = 8):
        self.path = Path(cache_path)
        self.workers = workers
        self.cache: dict[str, float] = {}
        self.tokens = 0
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.cache[d["k"]] = d["p"]

    def _ask(self, path: str, concern: dict, comment: dict) -> float:
        state = {"file": path,
                 "maintainer_comment": {"line": concern.get("line"), "text": str(concern.get("body", ""))[:1500]},
                 "ai_comment": {"line": comment.get("line"), "text": str(comment.get("body", ""))[:1500]}}
        body = json.dumps({"model": JEV_MODEL, "state": state, "questions": {"same": {
            "type": "noul", "instructions": QUESTION, "criteria": {"true": YES, "false": NO}}}}).encode()
        req = urllib.request.Request(JEV_URL, data=body, headers={
            "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}", "Content-Type": "application/json"})
        err: Exception | None = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    d = json.load(resp)
                with self._lock:
                    self.tokens += (d.get("usage") or {}).get("input_tokens", 0)
                return float(d["answers"]["same"]["noul"])
            except Exception as e:  # noqa: BLE001 - transient network/429: back off and retry
                err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Jev request failed: {err}")

    def probabilities(self, pairs: list[tuple[str, dict, dict]]) -> list[float]:
        """pairs: (file path, maintainer concern {line, body}, agent comment {line, body})."""
        keys = [_key(*p) for p in pairs]
        todo = {k: p for k, p in zip(keys, pairs) if k not in self.cache}
        if todo:
            with ThreadPoolExecutor(self.workers) as ex:
                answers = list(ex.map(lambda p: self._ask(*p), todo.values()))
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                for k, p in zip(todo, answers):
                    self.cache[k] = p
                    fh.write(json.dumps({"k": k, "p": p}) + "\n")
        return [self.cache[k] for k in keys]

    def concern_scores(self, concerns: list[dict], output: dict | None) -> list[float]:
        """For each maintainer concern, the best P(same concern) over the agent's comments on that file (0 if none)."""
        mine = [c for c in (output or {}).get("comments") or [] if isinstance(c, dict) and c.get("body")]
        pairs, owner = [], []
        for i, g in enumerate(concerns):
            for c in mine:
                if c.get("path") == g["path"]:
                    pairs.append((g["path"], g, c))
                    owner.append(i)
        best = [0.0] * len(concerns)
        for i, p in zip(owner, self.probabilities(pairs)):
            best[i] = max(best[i], p)
        return best


def missed_concerns(matcher: Matcher, task: Any, output: dict | None) -> list[dict]:
    """The maintainer's non-nit concerns the output did not raise (best P(same concern) < THRESHOLD)."""
    gold = [c for c in task.truth["concerns"] if not c["nit"]]
    return [g for g, p in zip(gold, matcher.concern_scores(gold, output)) if p < THRESHOLD]


def semantic_score(matcher: Matcher, task: Any, output: dict | None) -> float:
    """The concern-task score (`concerns.score_concerns`) with recall judged semantically: positives score
    semantic recall, discounted for comments beyond the budget; clean PRs are scored on silence as before."""
    from .concerns import COMMENT_BUDGET, score_concerns

    if output is None or not task.truth["positive"]:
        return score_concerns(task, output)[0]
    n = len([c for c in output.get("comments") or [] if isinstance(c, dict)])
    over = max(0, n - COMMENT_BUDGET)
    return semantic_metrics(matcher, task, output)["sem_recall"] * COMMENT_BUDGET / (COMMENT_BUDGET + over)


def semantic_metrics(matcher: Matcher, task: Any, output: dict | None) -> dict[str, Any]:
    """Recall of the maintainer's (non-nit) concerns judged semantically: `sem_recall` counts a concern as matched
    at P >= THRESHOLD, `soft_recall` is the mean best probability."""
    if not task.truth["positive"]:
        return {"sem_recall": None, "soft_recall": None}
    gold = [c for c in task.truth["concerns"] if not c["nit"]] or task.truth["concerns"]
    best = matcher.concern_scores(gold, output)
    return {"sem_recall": sum(p >= THRESHOLD for p in best) / len(best), "soft_recall": sum(best) / len(best)}
