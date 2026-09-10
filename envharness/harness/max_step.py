from envharness.core.envharness import EnvHarness
from envharness.core.envharness import ActionableEnv
from envharness.core.types import Action

class MaxStepHarness(EnvHarness):
    def __init__(self, inner: ActionableEnv, max_steps: int):
        super().__init__(inner)
        self.max_steps = max_steps
        self.current_steps = 0

    def reset(self, *args, **kwargs):
        self.current_steps = 0
        return super().reset(*args, **kwargs)

    def step(self, action: Action):
        self.current_steps += 1
        result = super().step(action)

        if not result.terminated and self.current_steps >= self.max_steps:
            result.truncated = True
        return result

    def save_state(self):
        return {
            "max_steps": self.max_steps,
            "current_steps": self.current_steps
        }

    @classmethod
    def from_state(cls, state: dict, inner = None) -> "MaxStepHarness":
        harness = cls(inner=inner, max_steps=state["max_steps"])
        harness.current_steps = state["current_steps"]

        return harness