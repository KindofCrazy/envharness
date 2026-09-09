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