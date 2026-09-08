from envharness.core.actionable_env import ActionableEnv
from envharness.core.types import Action, Observation, EnvResponse, EvaluationResult
from envharness.bridges.toy24.game import Toy24State, combine, reset_numbers, stop
from envharness.bridges.toy24.tools import Combine, Reset, Stop
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

        if action.name not in [tool.name for tool in self.tool_registry]:
            return EnvResponse(
                observation=self.observe(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "error": "unknown_action"
                }
            )
        

        for tool in self.tool_registry:
            if tool.name == action.name:
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
            info=info
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
        }

