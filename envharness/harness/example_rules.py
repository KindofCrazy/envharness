from envharness.harness.rules import Rules
from envharness.core.types import Blocked

class NoResetRules(Rules):
    def filter_action(self, action):
        if action.name == "reset":
            return Blocked(
                reason="Blocked Action Reset"
            )
        return super().filter_action(action)

class MaxStepRules(Rules):
    def __init__(self, inner, max_steps: int):
        super().__init__(inner)
        self.max_steps = max_steps
        self.currnt_steps = 0

    def reset(self, *args, **kwargs):
        self.currnt_steps = 0
        return super().reset(*args, **kwargs)

    def filter_action(self, action):
        self.currnt_steps += 1
        return super().filter_action(action)

    def modify_transition(self, action, response):
        if not response.terminated and self.currnt_steps >= self.max_steps:
            response.truncated = True

        return response