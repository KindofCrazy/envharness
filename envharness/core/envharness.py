from abc import abstractmethod
from envharness.core.actionable_env import ActionableEnv
from envharness.core.types import Action, Observation, EnvResponse, EvaluationResult, EnvResetResponse

class EnvHarness(ActionableEnv):
    def __init__(self, inner: ActionableEnv):
        self.inner = inner

    def reset(self, *args, **kwargs) -> EnvResetResponse:
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

    @classmethod
    def harness_type(cls) -> str:
        raise NotImplementedError(f"{cls.__name__} has no harness_type")

    def notify_replay_complete(self):
        return self.inner.notify_replay_complete()

    def default_reset_args(self):
        return self.inner.default_reset_args()

    def reset_after_load(self):
        return self.inner.reset_after_load()

    def close(self):
        return self.inner.close()
