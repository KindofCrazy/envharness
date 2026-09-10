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
