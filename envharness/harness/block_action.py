from envharness.core.envharness import EnvHarness
from envharness.core.types import EnvResponse, Action

class BlockActionHarness(EnvHarness):
    def __init__(self, inner, block_action: str):
        super().__init__(inner)
        self.block_action = block_action

    def step(self, action: Action) -> EnvResponse:
        if action.name != self.block_action:
            return super().step(action)
        else:
            return EnvResponse(
                observation=self.observe(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "blocked": True,
                    "reason": f"Blocked Action {self.block_action}"
                }
            )

    def save_state(self):
        return {
            "block_action": self.block_action
        }

    @classmethod
    def from_state(cls, state, inner = None) -> "BlockActionHarness":
        return cls(inner=inner, block_action=state["block_action"])
    