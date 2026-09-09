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
                    "reason": f"Blocked Actoin {self.block_action}"
                }
            )
