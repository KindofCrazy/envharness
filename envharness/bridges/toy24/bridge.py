from envharness.core.actionable_env import ActionableEnv
from envharness.core.types import Action, Observation, EnvResponse, EvaluationResult
from envharness.bridges.toy24.game import Toy24State
from envharness.bridges.toy24.tools import Combine, Reset, Stop
from envharness.core.registry import register_env

@register_env("toy24")
class Toy24Env(ActionableEnv):

    tool_registry = [Combine, Reset, Stop]

    def __init__(self):
        self.state = Toy24State()

    def reset(self, numbers, target=24):
        self.state = Toy24State(
            target=target,
            initial_numbers=list(numbers),
            current_numbers=[float(n) for n in numbers]
        )

        return self.observe()

    def step(self, action: Action):
        self.state.step_count += 1

        tool = next(
            (
                tool
                for tool in self.tool_registry
                if tool.name == action.name
            ),
            None,
        )

        if tool is None:
            return EnvResponse(
                observation=self.observe(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "error": "unknown_action"
                }
            )
        else:
            try:
                info=tool.invoke(self.state, **action.kwargs)
            except TypeError:
                return EnvResponse(
                    observation=self.observe(),
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info={
                        "error": "bad args" 
                    }
                )
        
        return EnvResponse(
            observation=self.observe(),
            reward=1.0 if self.state.stopped and self.state.success else 0.0,
            terminated=self.state.stopped,
            truncated=False,
            info={
                "result": info
            }
        )

    def observe(self):
        return Observation(
            text=f"target={self.state.target}, numbers={self.state.current_numbers}, history={self.state.history}, step_count={self.state.step_count}",
            data={
                "numbers": list(self.state.current_numbers),
                "target": self.state.target,
                "history": list(self.state.history),
                "step_count": self.state.step_count
            }
        )

    def evaluate(self):
        return EvaluationResult(
            success=self.state.stopped and self.state.success,
            score=1.0 if self.state.stopped and self.state.success else 0.0,
            metrics={
                "steps": self.state.step_count,
            }
        )

    def get_env_state(self) -> Toy24State:
        return self.state

    def save_state(self) -> dict:
        return {
            "target": self.state.target,
            "initial_numbers": list(self.state.initial_numbers),
            "current_numbers": list(self.state.current_numbers),
            "history": list(self.state.history),
            "stopped": self.state.stopped,
            "success": self.state.success,
            "step_count": self.state.step_count,
        }

    @classmethod
    def from_state(cls, state) -> "Toy24Env":
        env = cls()

        env.state = Toy24State(
            target=state["target"],
            initial_numbers=list(state["initial_numbers"]),
            current_numbers=list(state["current_numbers"]),
            history=list(state["history"]),
            stopped=state["stopped"],
            success=state["success"],
            step_count=state["step_count"],
        )

        return env
