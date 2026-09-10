from envharness.core.envharness import EnvHarness
from envharness.core.actionable_env import ActionableEnv
from envharness.core.types import EnvResponse, EvaluationResult

class Link(EnvHarness):
    def __init__(self, env_a: ActionableEnv, env_b: ActionableEnv):
        super().__init__(env_a)
        self.env_a = env_a
        self.env_b = env_b
        self.stage = "A"
        self.a_success = False
        self.b_success = False

    def reset(self, *args, **kwargs):
        self.stage = "A"
        self.a_success = False
        self.b_success = False
        return super().reset(*args, **kwargs)

    def step(self, action):
        if self.stage == "A":
            response = self.inner.step(action)
            if response.truncated or response.terminated:
                a_evaluation = super().evaluate()
                self.a_success = a_evaluation.success
                self.stage = "B"
                self.env_b.reset()
                return EnvResponse(
                    observation=self.observe(),
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info={
                        "link_stage": "B",
                        "a_success": self.a_success
                    }
                )
            else:
                return response
        else:
            response = self.env_b.step(action)
            if response.truncated or response.terminated:
                b_evaluation = self.env_b.evaluate()
                self.b_success = b_evaluation.success
                combined = self.a_success and self.b_success
                return EnvResponse(
                    observation=self.observe(),
                    reward=float(combined),
                    terminated=True,
                    truncated=False,
                    info={
                        "a_success": self.a_success,
                        "b_success": self.b_success,
                        "combined_success": combined
                    }
                )
            else:
                return response

    def evaluate(self):
        return EvaluationResult(
            success = self.a_success and self.b_success,
            score = 1.0 if self.a_success and self.b_success else 0.0
        )

    def observe(self):
        if self.stage == "A":
            return self.inner.observe()
        return self.env_b.observe()

    def save_state(self):
        raise NotImplementedError(
            "Link persistence is not supported"
        )

    @classmethod
    def from_state(cls, state, inner = None):
        raise NotImplementedError(
            "Link persistence is not supported"
        )
