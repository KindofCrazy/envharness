from typing import Any
from abc import ABC, abstractmethod

from envharness.core.types import Action, Observation, EnvResponse, EvaluationResult

class ActionableEnv(ABC):
    @abstractmethod
    def reset(self, *args, **kwargs) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> EnvResponse:
        ...

    @abstractmethod
    def observe(self) -> Observation:
        ...

    @abstractmethod
    def evaluate(self) -> EvaluationResult:
        ...

    @abstractmethod
    def get_env_state(self) -> Any:
        ...

    def save_state(self) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement save_state()"
        )

    def from_state(cls, state: dict) -> "ActionableEnv":
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_state()"
        )
