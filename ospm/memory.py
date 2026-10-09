"""Persistent shared memory (SQLite).

Holds everything that makes the system *self-evolving* across runs:

* episodes     – every decision an agent made, plus the eventual ground-truth outcome (feedback),
* cases        – labelled experience used for retrieval (k-NN) and as evolution training data,
* genomes      – every genome ever created, with lineage,
* deployments  – which genome is live for each agent, and the promotion history,
* rule_stats   – online credit assignment for individual playbook rules,
* llm_cache    – content-addressed LLM response cache (makes re-evaluation free).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .genome import Genome

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    agent TEXT NOT NULL,
    item_id TEXT NOT NULL,
    genome_id TEXT,
    item_text TEXT,
    item_json TEXT,
    decision_json TEXT NOT NULL,
    outcome_json TEXT,
    outcome_ts REAL
);
CREATE INDEX IF NOT EXISTS ix_episodes_agent_item ON episodes(agent, item_id);
CREATE TABLE IF NOT EXISTS cases (
    agent TEXT NOT NULL,
    case_id TEXT NOT NULL,
    text TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    item_json TEXT,
    source TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (agent, case_id)
);
CREATE TABLE IF NOT EXISTS genomes (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    json TEXT NOT NULL,
    origin TEXT,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS deployments (
    agent TEXT PRIMARY KEY,
    genome_id TEXT NOT NULL,
    ts REAL NOT NULL,
    metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS promotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    agent TEXT NOT NULL,
    genome_id TEXT NOT NULL,
    previous_genome_id TEXT,
    metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS evolution_runs (
    run_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    ts REAL NOT NULL,
    config_json TEXT,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS lineage (
    run_id TEXT NOT NULL,
    genome_id TEXT NOT NULL,
    generation INTEGER,
    operator TEXT,
    parents TEXT,
    minibatch_score REAL,
    train_score REAL,
    status TEXT
);
CREATE TABLE IF NOT EXISTS rule_stats (
    agent TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    fired INTEGER NOT NULL DEFAULT 0,
    correct INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent, rule_id)
);
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    ts REAL NOT NULL
);
"""


class Memory:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------ episodes
    def record_episode(self, agent: str, item_id: str, item_text: str, decision: dict, genome_id: str = "",
                       item: dict | None = None) -> int:
        cur = self._exec(
            "INSERT INTO episodes(ts, agent, item_id, genome_id, item_text, item_json, decision_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (time.time(), agent, item_id, genome_id, item_text, json.dumps(item) if item else None,
             json.dumps(decision, default=str)),
        )
        return int(cur.lastrowid)

    def latest_episode(self, agent: str, item_id: str) -> dict | None:
        rows = self._query(
            "SELECT * FROM episodes WHERE agent=? AND item_id=? ORDER BY id DESC LIMIT 1", (agent, item_id)
        )
        return self._episode(rows[0]) if rows else None

    def set_outcome(self, episode_id: int, outcome: dict) -> None:
        self._exec(
            "UPDATE episodes SET outcome_json=?, outcome_ts=? WHERE id=?",
            (json.dumps(outcome), time.time(), episode_id),
        )

    def episodes(self, agent: str | None = None, with_outcome: bool | None = None, limit: int = 1000) -> list[dict]:
        sql, params = "SELECT * FROM episodes WHERE 1=1", []
        if agent:
            sql += " AND agent=?"
            params.append(agent)
        if with_outcome is True:
            sql += " AND outcome_json IS NOT NULL"
        elif with_outcome is False:
            sql += " AND outcome_json IS NULL"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [self._episode(r) for r in self._query(sql, tuple(params))]

    @staticmethod
    def _episode(r: sqlite3.Row) -> dict:
        return {
            "id": r["id"],
            "ts": r["ts"],
            "agent": r["agent"],
            "item_id": r["item_id"],
            "genome_id": r["genome_id"],
            "item_text": r["item_text"],
            "item": json.loads(r["item_json"]) if r["item_json"] else None,
            "decision": json.loads(r["decision_json"]),
            "outcome": json.loads(r["outcome_json"]) if r["outcome_json"] else None,
        }

    # ------------------------------------------------------------------ cases
    def upsert_case(self, agent: str, case_id: str, text: str, labels: dict, source: str = "feedback",
                    item: dict | None = None) -> None:
        self._exec(
            "INSERT INTO cases(agent, case_id, text, labels_json, item_json, source, ts) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(agent, case_id) DO UPDATE SET text=excluded.text, labels_json=excluded.labels_json, "
            "item_json=COALESCE(excluded.item_json, cases.item_json), source=excluded.source, ts=excluded.ts",
            (agent, case_id, text, json.dumps(labels, sort_keys=True), json.dumps(item) if item else None, source,
             time.time()),
        )

    def cases(self, agent: str, source: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM cases WHERE agent=?", [agent]
        if source:
            sql += " AND source=?"
            params.append(source)
        sql += " ORDER BY case_id"
        return [
            {"id": r["case_id"], "text": r["text"], "labels": json.loads(r["labels_json"]), "source": r["source"],
             "item": json.loads(r["item_json"]) if r["item_json"] else None}
            for r in self._query(sql, tuple(params))
        ]

    # ------------------------------------------------------------------ genomes
    def save_genome(self, genome: Genome) -> str:
        gid = genome.id
        self._exec(
            "INSERT OR IGNORE INTO genomes(id, agent, json, origin, ts) VALUES (?,?,?,?,?)",
            (gid, genome.agent, genome.to_json(), genome.origin, time.time()),
        )
        return gid

    def load_genome(self, genome_id: str) -> Genome | None:
        rows = self._query("SELECT json FROM genomes WHERE id=?", (genome_id,))
        return Genome.from_json(rows[0]["json"]) if rows else None

    def genomes(self, agent: str) -> list[Genome]:
        return [Genome.from_json(r["json"]) for r in self._query("SELECT json FROM genomes WHERE agent=? ORDER BY ts", (agent,))]

    def deploy(self, genome: Genome, metrics: dict | None = None) -> None:
        """Make `genome` the live genome for its agent and append to the promotion history."""
        self.save_genome(genome)
        prev = self.deployed_id(genome.agent)
        with self._lock:
            self._conn.execute(
                "INSERT INTO deployments(agent, genome_id, ts, metrics_json) VALUES (?,?,?,?) "
                "ON CONFLICT(agent) DO UPDATE SET genome_id=excluded.genome_id, ts=excluded.ts, "
                "metrics_json=excluded.metrics_json",
                (genome.agent, genome.id, time.time(), json.dumps(metrics or {})),
            )
            self._conn.execute(
                "INSERT INTO promotions(ts, agent, genome_id, previous_genome_id, metrics_json) VALUES (?,?,?,?,?)",
                (time.time(), genome.agent, genome.id, prev, json.dumps(metrics or {})),
            )
            self._conn.commit()

    def deployed_id(self, agent: str) -> str | None:
        rows = self._query("SELECT genome_id FROM deployments WHERE agent=?", (agent,))
        return rows[0]["genome_id"] if rows else None

    def deployed(self, agent: str) -> Genome | None:
        gid = self.deployed_id(agent)
        return self.load_genome(gid) if gid else None

    def deployment_metrics(self, agent: str) -> dict:
        rows = self._query("SELECT metrics_json FROM deployments WHERE agent=?", (agent,))
        return json.loads(rows[0]["metrics_json"] or "{}") if rows else {}

    def promotions(self, agent: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM promotions", ()
        if agent:
            sql, params = sql + " WHERE agent=?", (agent,)
        return [
            {
                "ts": r["ts"],
                "agent": r["agent"],
                "genome_id": r["genome_id"],
                "previous_genome_id": r["previous_genome_id"],
                "metrics": json.loads(r["metrics_json"] or "{}"),
            }
            for r in self._query(sql + " ORDER BY id", params)
        ]

    # ------------------------------------------------------------------ evolution bookkeeping
    def start_run(self, run_id: str, agent: str, config: dict) -> None:
        self._exec(
            "INSERT INTO evolution_runs(run_id, agent, ts, config_json) VALUES (?,?,?,?)",
            (run_id, agent, time.time(), json.dumps(config, default=str)),
        )

    def finish_run(self, run_id: str, summary: dict) -> None:
        self._exec("UPDATE evolution_runs SET summary_json=? WHERE run_id=?", (json.dumps(summary, default=str), run_id))

    def runs(self, agent: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM evolution_runs", ()
        if agent:
            sql, params = sql + " WHERE agent=?", (agent,)
        return [
            {
                "run_id": r["run_id"],
                "agent": r["agent"],
                "ts": r["ts"],
                "config": json.loads(r["config_json"] or "{}"),
                "summary": json.loads(r["summary_json"]) if r["summary_json"] else None,
            }
            for r in self._query(sql + " ORDER BY ts", params)
        ]

    def log_lineage(
        self,
        run_id: str,
        genome: Genome,
        operator: str,
        minibatch_score: float | None,
        train_score: float | None,
        status: str,
    ) -> None:
        self._exec(
            "INSERT INTO lineage(run_id, genome_id, generation, operator, parents, minibatch_score, train_score, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run_id, genome.id, genome.generation, operator, json.dumps(genome.parents), minibatch_score, train_score, status),
        )

    def lineage(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._query("SELECT * FROM lineage WHERE run_id=? ORDER BY rowid", (run_id,))]

    # ------------------------------------------------------------------ rule credit assignment
    def update_rule_stats(self, agent: str, rule_id: str, correct: bool) -> None:
        self._exec(
            "INSERT INTO rule_stats(agent, rule_id, fired, correct) VALUES (?,?,1,?) "
            "ON CONFLICT(agent, rule_id) DO UPDATE SET fired=fired+1, correct=correct+excluded.correct",
            (agent, rule_id, int(correct)),
        )

    def rule_stats(self, agent: str) -> dict[str, tuple[int, int]]:
        rows = self._query("SELECT rule_id, fired, correct FROM rule_stats WHERE agent=?", (agent,))
        return {r["rule_id"]: (r["fired"], r["correct"]) for r in rows}

    # ------------------------------------------------------------------ LLM cache (ResponseCache protocol)
    def get(self, key: str) -> dict | None:
        rows = self._query("SELECT value_json FROM llm_cache WHERE key=?", (key,))
        return json.loads(rows[0]["value_json"]) if rows else None

    def put(self, key: str, value: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO llm_cache(key, value_json, ts) VALUES (?,?,?)", (key, json.dumps(value), time.time())
        )

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for table in ("episodes", "cases", "genomes", "promotions", "evolution_runs", "lineage", "llm_cache"):
            out[table] = self._query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
        out["episodes_with_outcome"] = self._query("SELECT COUNT(*) AS n FROM episodes WHERE outcome_json IS NOT NULL")[0]["n"]
        out["deployed"] = {r["agent"]: r["genome_id"] for r in self._query("SELECT agent, genome_id FROM deployments")}
        return out
