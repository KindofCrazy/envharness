from dataclasses import dataclass, field
from typing import Any, Literal
from enum import Enum

@dataclass
class Action:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class Blocked:
    reason: str = ""

@dataclass
class Observation:
    text: str
    data: dict[str, Any] = field(default_factory=dict)

@dataclass
class EnvResponse:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    success: bool
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

@dataclass
class Step:
    observation: Observation
    action: Action
    response: EnvResponse

class TraceKind(str, Enum):
    BASELINE = "baseline"
    EXPLORATION = "exploration"
    ACCEPTED = "accepted"

@dataclass
class Trace:
    initial_observation: Observation
    steps: list[Step] = field(default_factory=list)
    success: bool = False

    kind: TraceKind = TraceKind.EXPLORATION

    task_id: int | str | None = None
    attempt_idx: int | None = None
    rollout_idx: int | None = None

@dataclass
class Candidate:
    rules_code: str = ""
    in_env_actions: list[Action] = field(default_factory=list)

@dataclass
class Diagnosis:
    summary: str

@dataclass
class DesignProposal:
    diagnosis: Diagnosis
    candidate: Candidate

@dataclass
class ValidationBatch:
    traces: list[Trace] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(trace.success for trace in self.traces)

    @property
    def success_rate(self) -> float:
        if not self.traces:
            return 0.0
        return self.success_count / len(self.traces)

@dataclass
class ValidationComparison:
    baseline: ValidationBatch
    candidate: ValidationBatch

    @property
    def delta_success_rate(self) -> float:
        return self.candidate.success_rate - self.baseline.success_rate

@dataclass
class ObjectiveResult:
    satisfied: bool
    score: float = 0.0
    diagnostic: str = ""
    suggestion: str = ""

@dataclass
class ObjectiveSignal:
    score: float
    diagnostic: str = ""
    suggestion: str = ""

@dataclass
class ValidationAttempt:
    proposal: DesignProposal
    validation: ValidationBatch
    comparison: ValidationComparison
    objective_result: ObjectiveResult

@dataclass
class DesignIterationResult:
    baseline: ValidationBatch
    proposal_trace: Trace
    attempts: list[ValidationAttempt] = field(default_factory=list)
    accepted: bool = False

class Decision(str, Enum):
    ACCEPT = "accept"
    REFINE = "refine"
    REJECT = "reject"

@dataclass
class FailureAnalysis:
    primary_axis: (
        Literal["S0", "A", "O", "T", "R", "task_understanding", "none"] | None
    ) = None

    label: str = ""
    description: str = ""

@dataclass
class DecideResult:
    decision: Decision
    failure_analysis: FailureAnalysis | None = None
    rationale: str = ""

@dataclass
class BaselineRolloutSummary:
    success: bool
    steps: int

@dataclass
class BaselineSnapshot:
    n: int
    n_success: int
    success_rate: float
    avg_success_steps: float | None = None
    per_rollout: list[BaselineRolloutSummary] = field(default_factory=list)

