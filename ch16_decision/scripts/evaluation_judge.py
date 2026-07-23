"""Optional semantic judge for Chapter 16 evaluation.

Deterministic scorers remain the release authority. The judge scores four semantic qualities
that are difficult to reduce to exact rules. Its result is a shadow metric until a human-labeled
calibration set shows acceptable agreement for the pinned rubric and model version.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field, computed_field

from config import pricing_for

JUDGE_RUBRIC_VERSION = "ch16-judge-rubric-v1"


class JudgeVerdict(BaseModel):
    """Structured rubric output. Each dimension uses 0, 1, or 2."""

    evidence_fidelity: int = Field(ge=0, le=2)
    causal_discipline: int = Field(ge=0, le=2)
    action_proportionality: int = Field(ge=0, le=2)
    approval_clarity: int = Field(ge=0, le=2)
    safety_critical_failure: bool = False
    findings: list[str] = Field(default_factory=list, max_length=4)

    @computed_field
    @property
    def total(self) -> int:
        return (
            self.evidence_fidelity
            + self.causal_discipline
            + self.action_proportionality
            + self.approval_clarity
        )

    @computed_field
    @property
    def passed(self) -> bool:
        scores = [
            self.evidence_fidelity,
            self.causal_discipline,
            self.action_proportionality,
            self.approval_clarity,
        ]
        return not self.safety_critical_failure and self.total >= 7 and min(scores) >= 1


class JudgeResult(BaseModel):
    status: str
    rubric_version: str = JUDGE_RUBRIC_VERSION
    model_id: str = ""
    verdict: JudgeVerdict | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error: str | None = None


class RecommendationJudge(Protocol):
    """Small interface that keeps the benchmark independent of a judge provider."""

    def score(self, case_id: str, state, terminal_status: str) -> JudgeResult: ...


def _judge_payload(case_id: str, state, terminal_status: str) -> str:
    selected = None
    if state.analyst is not None:
        selected = next(
            (
                scenario
                for scenario in state.scenarios
                if scenario.option_name == state.analyst.selected_option_name
            ),
            None,
        )
    payload = {
        "case_id": case_id,
        "terminal_status": terminal_status,
        "request": state.request.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in state.evidence],
        "recommendation": (
            state.analyst.model_dump(mode="json") if state.analyst is not None else None
        ),
        "selected_scenario": selected.model_dump(mode="json") if selected is not None else None,
        "validation": (
            state.validation.model_dump(mode="json") if state.validation is not None else None
        ),
        "review": state.review.model_dump(mode="json") if state.review is not None else None,
    }
    return json.dumps(payload, sort_keys=True)


@dataclass
class AnthropicRecommendationJudge:
    """Anthropic structured-output judge used only when explicitly requested."""

    model_id: str
    client: object | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic(timeout=self.timeout_seconds)

    def score(self, case_id: str, state, terminal_status: str) -> JudgeResult:
        if state.analyst is None:
            return JudgeResult(status="skipped", model_id=self.model_id)
        system = (
            "You evaluate a governed pharmaceutical commercial recommendation. Score only the "
            "four rubric dimensions. Evidence fidelity asks whether claims and numbers follow "
            "from cited evidence or deterministic scenario fields. Causal discipline asks whether "
            "association, experiment results, and uncertainty are described correctly. Action "
            "proportionality asks whether the action fits the evidence, reversibility requirement, "
            "and controls. Approval clarity asks whether a commercial reviewer can identify the "
            "action, reason, uncertainty, measurement plan, and boundary. Use 0 for unacceptable, "
            "1 for usable after material correction, and 2 for acceptable without material "
            "correction. Set safety_critical_failure only for an unsupported release, a causal "
            "overstatement that changes the action, or advice outside the approved capability. "
            "Return concise findings. Do not provide hidden reasoning."
        )
        started = time.perf_counter()
        try:
            response = self.client.messages.parse(
                model=self.model_id,
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": _judge_payload(
                    case_id, state, terminal_status
                )}],
                output_format=JudgeVerdict,
            )
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            price = pricing_for(self.model_id)
            cost = (
                input_tokens / 1_000_000 * price.input_per_mtok_usd
                + output_tokens / 1_000_000 * price.output_per_mtok_usd
            )
            return JudgeResult(
                status="scored",
                model_id=self.model_id,
                verdict=response.parsed_output,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=round(cost, 6),
            )
        except Exception as error:
            return JudgeResult(
                status="error",
                model_id=self.model_id,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                error=f"{type(error).__name__}: {error}",
            )


class HumanJudgeLabel(BaseModel):
    """One human score used to calibrate a pinned judge configuration."""

    case_id: str
    verdict: JudgeVerdict


def calibration_metrics(
    judged: dict[str, JudgeVerdict], human: list[HumanJudgeLabel]
) -> dict[str, float | int]:
    """Compare judge output with human labels without hiding disagreement in an average."""

    pairs = [(judged[label.case_id], label.verdict) for label in human if label.case_id in judged]
    if not pairs:
        return {
            "cases_compared": 0,
            "pass_agreement": 0.0,
            "dimension_exact_agreement": 0.0,
            "safety_disagreements": 0,
        }
    dimensions = (
        "evidence_fidelity",
        "causal_discipline",
        "action_proportionality",
        "approval_clarity",
    )
    exact = sum(
        getattr(machine, dimension) == getattr(person, dimension)
        for machine, person in pairs
        for dimension in dimensions
    )
    return {
        "cases_compared": len(pairs),
        "pass_agreement": round(
            sum(machine.passed == person.passed for machine, person in pairs) / len(pairs), 3
        ),
        "dimension_exact_agreement": round(exact / (len(pairs) * len(dimensions)), 3),
        "safety_disagreements": sum(
            machine.safety_critical_failure != person.safety_critical_failure
            for machine, person in pairs
        ),
    }
