from abc import ABC, abstractmethod

from envharness.core.types import Action, Observation, EnvResponse, EvaluationResult

class ActionableEnv(ABC):
    @abstractmethod
    def reset(self) -> EnvResponse:
        ...

    @abstractmethod
    def step(self, action: Action) -> EnvResponse:
        ...

    @abstractmethod
    def observe() -> Observation:
        ...

    def evaluate() -> EvaluationResult:
        ...
