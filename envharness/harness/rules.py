from envharness.core.envharness import EnvHarness
from envharness.core.types import Action, Blocked, Observation, EnvResponse

class Rules(EnvHarness):
    
    rules_code: str = ""

    def __init__(self, inner):
        super().__init__(inner)

    def filter_action(self, action: Action) -> Action | Blocked:
        return action

    def modify_transition(self, action: Action, response: EnvResponse) -> EnvResponse:
        return response

    def filter_observation(self, observation: Observation) -> Observation:
        return observation

    def step(self, action: Action) -> EnvResponse:
        filtered = self.filter_action(action)

        if isinstance(filtered, Blocked):
            return EnvResponse(
                observation=self.observe(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "reason": filtered.reason
                }
            )
        else:
            response = super().step(filtered)
            response = self.modify_transition(filtered, response)
            response.observation = self.filter_observation(response.observation)

        return response

    def observe(self) -> Observation:
        observation = super().observe()
        return self.filter_observation(observation)

    def reset(self, *args, **kwargs) -> Observation:
        observation = super().reset(*args, **kwargs)
        return self.filter_observation(observation)

    def save_state(self) -> dict:
        return {"rules_code": self.rules_code}

    @classmethod
    def from_state(cls, state, inner = None) -> "Rules":
        code = state.get("rules_code", "").strip()
        if not code:
            return cls(inner=inner)

        from envharness.core.code_loader import (
            load_rules_subclass,
        )


        RulesCls = load_rules_subclass(code)
        instance = RulesCls(inner=inner)
        instance.rules_code = code
        return instance
