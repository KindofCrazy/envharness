from dataclasses import dataclass, field
from typing import Any

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

@dataclass
class Trace:
    initial_observation: Observation
    steps: list[Step] = field(default_factory=list)
    success: bool = False

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
