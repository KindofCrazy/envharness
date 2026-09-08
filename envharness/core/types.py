from dataclasses import dataclass, Any, field

@dataclass
class Action:
    name: str = field(default_factory=str)
    kwargs: dict[str, Any] = field(default_factory=dict[str, Any])

@dataclass
class Observation:
    text: str
    data: dict[str, Any] = field(default_factory=dict)

@dataclass
class EnvResonse:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    success: bool
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
