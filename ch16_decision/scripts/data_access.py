"""Governed access to the Chapter 16 analytics environment.

The agents reach the database only through this layer: a read-only connection to the
approved file, a SQL guard that permits a single SELECT over approved objects, a row cap,
and provenance on every result. The hidden-truth file is never opened here, so no approved
query can reach the planted response truth.

This is a teaching control. A production system enforces the same boundary with database
roles, a query proxy, and network isolation rather than an in-process guard.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "assets" / "data"
ANALYTICS_DB = DATA_DIR / "analytics.duckdb"

ROW_CAP = 1000

APPROVED_OBJECTS = {
    "data_products", "hcp_dma_crosswalk", "hcp_digital_engagement", "dtc_dma_delivery",
    "closed_claims", "rx_weekly", "mmm_channel_results", "experiment_results",
    "primary_research", "market_events", "prior_decisions", "model_registry",
}

# Statement types and administrative commands the agent SQL role may never run.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|create|drop|alter|attach|detach|copy|pragma|install|load|"
    r"export|import|call|set|replace|vacuum|checkpoint|use)\b",
    re.IGNORECASE,
)
_IDENTIFIER_AFTER_SOURCE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)


class SqlNotAllowed(ValueError):
    """Raised when a query violates the read-only, approved-object policy."""


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    row_count: int
    elapsed_ms: float
    accessed_objects: list[str]
    result_hash: str = ""


# Audit trail: every governed query appends a record here, so a run can be inspected and a
# result reproduced. In a full system this is persisted with the decision state.
QUERY_LOG: list["QueryRecord"] = []


@dataclass
class QueryRecord:
    sql: str
    accessed_objects: list[str]
    row_count: int
    elapsed_ms: float
    result_hash: str


def guard_sql(sql: str) -> str:
    """Validate and normalize a query. Returns the safe SQL to run, or raises SqlNotAllowed."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise SqlNotAllowed("Empty query.")
    if ";" in text:
        raise SqlNotAllowed("Only a single statement is allowed.")
    head = text.lstrip("(").lower()
    if not (head.startswith("select") or head.startswith("with")):
        raise SqlNotAllowed("Only SELECT or WITH ... SELECT queries are allowed.")
    if _FORBIDDEN.search(text):
        raise SqlNotAllowed("Query contains a forbidden write or administrative command.")
    referenced = {name.lower() for name in _IDENTIFIER_AFTER_SOURCE.findall(text)}
    unknown = referenced - APPROVED_OBJECTS
    if unknown:
        raise SqlNotAllowed(f"Query references non-approved objects: {sorted(unknown)}")
    if not re.search(r"\blimit\b", text, re.IGNORECASE):
        text = f"{text}\nLIMIT {ROW_CAP}"
    return text


def query_approved_data(sql: str) -> QueryResult:
    """Run a guarded, read-only query against the approved analytics database."""
    safe = guard_sql(sql)
    con = duckdb.connect(str(ANALYTICS_DB), read_only=True)
    try:
        start = time.perf_counter()
        cur = con.execute(safe)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description]
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        con.close()
    accessed = sorted({n.lower() for n in _IDENTIFIER_AFTER_SOURCE.findall(safe)})
    result_hash = hashlib.sha256(repr(rows).encode()).hexdigest()[:12]
    QUERY_LOG.append(QueryRecord(safe, accessed, len(rows), round(elapsed_ms, 1), result_hash))
    return QueryResult(safe, columns, rows, len(rows), round(elapsed_ms, 1), accessed, result_hash)


def inspect_data_catalog() -> QueryResult:
    """Return the metadata catalog: what data products exist, their level, freshness, and use."""
    return query_approved_data(
        "SELECT data_product, entity_level, refresh_date, completeness, permitted_use "
        "FROM data_products ORDER BY data_product"
    )


def approved_schema() -> dict[str, list[str]]:
    """Column names for each approved object, so an agent can write valid SQL against them."""
    con = duckdb.connect(str(ANALYTICS_DB), read_only=True)
    try:
        schema = {}
        for obj in sorted(APPROVED_OBJECTS):
            cols = con.execute(f"PRAGMA table_info('{obj}')").fetchall()
            schema[obj] = [c[1] for c in cols]
        return schema
    finally:
        con.close()


def schema_digest() -> str:
    """A compact one-line-per-table schema string for a prompt."""
    return "\n".join(f"{table}({', '.join(cols)})" for table, cols in approved_schema().items())


def schema_hints() -> str:
    """Dialect and key categorical values so an agent writes SQL the environment accepts."""
    con = duckdb.connect(str(ANALYTICS_DB), read_only=True)
    try:
        segs = [r[0] for r in con.execute(
            "SELECT DISTINCT segment FROM hcp_dma_crosswalk").fetchall()]
        acc = [r[0] for r in con.execute(
            "SELECT DISTINCT access_state FROM hcp_dma_crosswalk").fetchall()]
        weeks = [r[0] for r in con.execute(
            "SELECT DISTINCT week FROM rx_weekly ORDER BY week").fetchall()]
    finally:
        con.close()
    return (
        f"This is DuckDB SQL. The 'week' column is a text label like '{weeks[0]}' "
        f"(values: {', '.join(weeks)}); do not use date functions on it. "
        f"segment values: {', '.join(segs)}. access_state values: {', '.join(acc)}. "
        "When tables contain repeated HCP-week or patient-level rows, pre-aggregate each "
        "source to the intended result level before joining. A join on HCP alone can multiply "
        "weekly rows. Describe summed NRx as observed NRx unless the query contains a causal "
        "comparison."
    )
