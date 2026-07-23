"""Durable case and run store for the Chapter 16 runtime.

Three forms of state (Section 16.5 of the chapter):

* Working state lives in the typed ``DecisionState`` passed between graph nodes.
* Durable checkpoint state is the LangGraph ``SqliteSaver`` checkpoint, so an interrupted
  run resumes from its pending node after a process restart.
* Decision and outcome history is structured storage: approved decisions, later outcomes, and
  run metadata keyed by a stable case ID, so the later run loads the earlier decision by ID
  rather than reconstructing it from a hardcoded summary.

This module owns the structured history in a local SQLite file. It is a teaching store; a
production system would use a managed database with the same typed records.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from models import (
    CaseRecord,
    OutcomeEvent,
    PriorDecisionRecord,
    RunMetadata,
    RunSummary,
)

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "assets" / "runtime"
CASE_STORE_PATH = RUNTIME_DIR / "case_store.sqlite"
CHECKPOINT_PATH = RUNTIME_DIR / "checkpoints.sqlite"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    metadata TEXT NOT NULL,
    state TEXT,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prior_decisions (
    case_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    case_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


class CaseStore:
    """Structured, durable storage for cases, runs, decisions, and outcomes."""

    def __init__(self, path: Path | str = CASE_STORE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False plus a lock so background run workers can write safely.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _write(self, sql: str, params: tuple) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _read_one(self, sql: str, params: tuple):
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _read_all(self, sql: str, params: tuple):
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # --- cases ---------------------------------------------------------------
    def save_case(self, case: CaseRecord) -> None:
        self._write(
            "INSERT OR REPLACE INTO cases (case_id, payload, created_at) VALUES (?, ?, ?)",
            (case.case_id, case.model_dump_json(), case.created_at),
        )

    def get_case(self, case_id: str) -> CaseRecord | None:
        row = self._read_one("SELECT payload FROM cases WHERE case_id = ?", (case_id,))
        return CaseRecord.model_validate_json(row[0]) if row else None

    def list_cases(self) -> list[CaseRecord]:
        rows = self._read_all("SELECT payload FROM cases ORDER BY created_at", ())
        return [CaseRecord.model_validate_json(row[0]) for row in rows]

    # --- runs ----------------------------------------------------------------
    def save_run(self, meta: RunMetadata, mode: str, state_json: str | None = None) -> None:
        self._write(
            "INSERT OR REPLACE INTO runs "
            "(run_id, case_id, mode, metadata, state, started_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                meta.run_id, meta.case_id, mode, meta.model_dump_json(),
                state_json, meta.started_at, meta.updated_at,
            ),
        )

    def get_run_meta(self, run_id: str) -> RunMetadata | None:
        row = self._read_one("SELECT metadata FROM runs WHERE run_id = ?", (run_id,))
        return RunMetadata.model_validate_json(row[0]) if row else None

    def get_run_mode(self, run_id: str) -> str | None:
        row = self._read_one("SELECT mode FROM runs WHERE run_id = ?", (run_id,))
        return row[0] if row else None

    def get_run_state(self, run_id: str) -> str | None:
        row = self._read_one("SELECT state FROM runs WHERE run_id = ?", (run_id,))
        return row[0] if row and row[0] else None

    def list_runs(self, case_id: str) -> list[RunSummary]:
        rows = self._read_all(
            "SELECT run_id, case_id, mode, metadata, started_at, updated_at "
            "FROM runs WHERE case_id = ? ORDER BY started_at",
            (case_id,),
        )
        summaries: list[RunSummary] = []
        for run_id, cid, mode, metadata, started_at, updated_at in rows:
            meta = RunMetadata.model_validate_json(metadata)
            summaries.append(RunSummary(
                run_id=run_id, case_id=cid, mode=mode, status=meta.status,
                current_node=meta.current_node, started_at=started_at, updated_at=updated_at,
            ))
        return summaries

    # --- decision and outcome history ---------------------------------------
    def save_prior_decision(self, record: PriorDecisionRecord) -> None:
        self._write(
            "INSERT OR REPLACE INTO prior_decisions (case_id, payload) VALUES (?, ?)",
            (record.case_id, record.model_dump_json()),
        )

    def get_prior_decision(self, case_id: str) -> PriorDecisionRecord | None:
        row = self._read_one(
            "SELECT payload FROM prior_decisions WHERE case_id = ?", (case_id,))
        return PriorDecisionRecord.model_validate_json(row[0]) if row else None

    def save_outcome(self, outcome: OutcomeEvent) -> None:
        self._write(
            "INSERT OR REPLACE INTO outcomes (case_id, payload) VALUES (?, ?)",
            (outcome.case_id, outcome.model_dump_json()),
        )

    def get_outcome(self, case_id: str) -> OutcomeEvent | None:
        row = self._read_one("SELECT payload FROM outcomes WHERE case_id = ?", (case_id,))
        return OutcomeEvent.model_validate_json(row[0]) if row else None
