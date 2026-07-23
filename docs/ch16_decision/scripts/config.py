"""Central configuration for Chapter 16: trigger thresholds, component versions, pricing.

Keeping the deterministic trigger thresholds, the component version strings, and the model
pricing snapshot in one place lets the version manifest, the signal monitor, and the run
metadata cite the same values. Change a threshold here rather than inside a node.
"""

from __future__ import annotations

from datetime import date

from models import PricingSnapshot, RuntimeLimits

# --- Component versions (recorded in every run and evaluation report, Section 26) ----------

TRIGGER_RULE_VERSION = "ch16-trigger-v1"
GRAPH_VERSION = "ch16-graph-v4"
TOOL_VERSION = "ch16-tools-v1"
DATA_VERSION = "ch16-data-seed16-v1"
SCHEMA_VERSION = "ch16-schema-v2"
PROMPT_VERSION = "ch16-prompts-v4"
BENCHMARK_VERSION = "ch16-benchmark-v3"

# --- Decision dates for the two-date Roventra case ----------------------------------------

FIRST_DECISION_DATE = date(2026, 7, 14)
LATER_DECISION_DATE = date(2026, 10, 6)

# --- Deterministic trigger rule (Section 19.2) --------------------------------------------

TRIGGER = {
    "brand": "Roventra",
    "metric": "community HCP digital clicks (weekly)",
    "population": "community endocrinologists",
    "geography": "US DMAs",
    "source": "hcp_digital_engagement x hcp_dma_crosswalk",
    # Community clicks must rise at least this fraction from first to last monitor week.
    "min_engagement_rise": 0.30,
    # A candidate opens only while NRx evidence is unsettled: claims below this maturity OR
    # aggregate weekly NRx growth no greater than the ceiling below.
    "claims_maturity_ceiling": 0.80,
    "max_nrx_growth": 0.05,
    "monitor_weeks": ["2026-W23", "2026-W24", "2026-W25", "2026-W26", "2026-W27"],
}

# --- Default runtime budgets and pricing --------------------------------------------------

DEFAULT_LIMITS = RuntimeLimits()

# Anthropic list pricing per million tokens (USD). Update the effective date when it changes.
PRICING = {
    "claude-haiku-4-5": PricingSnapshot(
        model_id="claude-haiku-4-5",
        input_per_mtok_usd=1.00,
        output_per_mtok_usd=5.00,
        effective_date="2026-07-01",
    ),
    "claude-sonnet-5": PricingSnapshot(
        model_id="claude-sonnet-5",
        input_per_mtok_usd=3.00,
        output_per_mtok_usd=15.00,
        effective_date="2026-07-01",
    ),
    "claude-opus-4-8": PricingSnapshot(
        model_id="claude-opus-4-8",
        input_per_mtok_usd=5.00,
        output_per_mtok_usd=25.00,
        effective_date="2026-07-01",
    ),
    # OpenRouter list pricing for Anthropic models is typically at parity with the direct
    # Anthropic rate above. Treat these as an estimate; confirm against the OpenRouter
    # dashboard for the exact current rate before relying on a cost report.
    "anthropic/claude-haiku-4.5": PricingSnapshot(
        model_id="anthropic/claude-haiku-4.5",
        input_per_mtok_usd=1.00,
        output_per_mtok_usd=5.00,
        effective_date="2026-07-01",
        pricing_version="ch16-pricing-v1-openrouter-estimate",
    ),
    "anthropic/claude-sonnet-5": PricingSnapshot(
        model_id="anthropic/claude-sonnet-5",
        input_per_mtok_usd=3.00,
        output_per_mtok_usd=15.00,
        effective_date="2026-07-01",
        pricing_version="ch16-pricing-v1-openrouter-estimate",
    ),
    "anthropic/claude-opus-4.8": PricingSnapshot(
        model_id="anthropic/claude-opus-4.8",
        input_per_mtok_usd=5.00,
        output_per_mtok_usd=25.00,
        effective_date="2026-07-01",
        pricing_version="ch16-pricing-v1-openrouter-estimate",
    ),
}


def pricing_for(model_id: str) -> PricingSnapshot:
    """Return the pricing snapshot for a model, falling back to a conservative default."""
    if model_id in PRICING:
        return PRICING[model_id]
    return PricingSnapshot(
        model_id=model_id,
        input_per_mtok_usd=5.00,
        output_per_mtok_usd=25.00,
        effective_date="2026-07-01",
        pricing_version="ch16-pricing-v1-fallback",
    )
