from envharness.core.envharness import EnvHarness
from envharness.core.types import Action

class Setup(EnvHarness):
    def __init__(self, inner, actions: list[Action]):
        super().__init__(inner)
        self.actions = actions

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        for action in self.actions:
            res = super().step(action)

            if res.terminated or res.truncated:
                break

        return self.observe()

