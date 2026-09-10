from abc import abstractmethod
from envharness.core.actionable_env import ActionableEnv
from envharness.core.types import Action, Observation, EnvResponse, EvaluationResult

class EnvHarness(ActionableEnv):
    def __init__(self, inner: ActionableEnv):
        self.inner = inner

    def reset(self, *args, **kwargs) -> Observation:
        return self.inner.reset(*args, **kwargs)

    def step(self, action: Action) -> EnvResponse:
        return self.inner.step(action)

    def observe(self) -> Observation:
        return self.inner.observe()

    def evaluate(self) -> EvaluationResult:
        return self.inner.evaluate()

    def get_env_state(self):
        return self.inner.get_env_state()

    @abstractmethod
    def save_state(self) -> dict:
        ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict, inner: ActionableEnv | None = None) -> "EnvHarness":
        ...
