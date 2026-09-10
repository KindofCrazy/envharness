from typing import Any
from abc import ABC, abstractmethod

from envharness.core.types import Action, Observation, EnvResponse, EvaluationResult

class ActionableEnv(ABC):

    tool_registry = []

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

    @classmethod
    def from_state(cls, state: dict) -> "ActionableEnv":
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_state()"
        )

    @classmethod
    def env_type(cls) -> str:
        raise NotImplementedError(f"{cls.__name__} has no env_type")

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        return [tool.get_info() for tool in cls.tool_registry]

    @classmethod
    def env_state_schema(cls) -> str:
        return "(no env_state schema declared)"