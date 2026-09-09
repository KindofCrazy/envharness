from envharness.core.actionable_env import ActionableEnv
from envharness.core.types import Action

class EnvHarness(ActionableEnv):
    def __init__(self, inner: ActionableEnv):
        super().__init__()
        self.inner = inner

    def reset(self, *args, **kwargs):
        return self.inner.reset(args, kwargs)

    def step(self, action: Action):
        return self.inner.step(action)

    def observe(self):
        return self.inner.observe()

    def evaluate(self):
        return self.inner.evaluate()

