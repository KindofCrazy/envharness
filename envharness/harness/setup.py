from envharness.core.envharness import EnvHarness
from envharness.core.types import Action, EnvResetResponse
from envharness.core.registry import register_harness
@register_harness("setup")
class Setup(EnvHarness):
    def __init__(self, inner, actions: list[Action]):
        super().__init__(inner)
        self.actions = actions

    def reset(self, *args, **kwargs):
        reset_response = self.inner.reset(*args, **kwargs)
        for action in self.actions:
            self.inner.step(action)

        if self.actions:
            self.inner.notify_replay_complete()

        return EnvResetResponse(
            observation=self.inner.observe(),
            info=dict(reset_response.info)
        )


    def save_state(self) -> dict:
        return {
            "actions": [
                {
                    "name": action.name,
                    "kwargs": dict(action.kwargs)
                } for action in self.actions
            ]
        }

    @classmethod
    def from_state(cls, state, inner = None) -> "Setup":
        actions = [
            Action(
                name=item["name"],
                kwargs=dict(item["kwargs"]),
            )
            for item in state["actions"]
        ]

        return cls(
            inner=inner,
            actions=actions,
        )
