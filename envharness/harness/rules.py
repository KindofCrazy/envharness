from envharness.core.envharness import EnvHarness
from envharness.core.types import Action, Blocked, Observation, EnvResponse
from envharness.core.registry import register_harness

@register_harness("rules")
class Rules(EnvHarness):
    
    rules_code: str = ""

    def __init__(self, inner):
        super().__init__(inner)

    def filter_action(self, action: Action, env_state) -> Action | Blocked:
        return action

    def modify_transition(self, action: Action, response: EnvResponse, env_state) -> EnvResponse:
        return response

    def filter_observation(self, observation: Observation, env_state) -> Observation:
        return observation

    def step(self, action: Action) -> EnvResponse:
        pre_state = self.inner.get_env_state()
        filtered = self.filter_action(action, pre_state)

        if isinstance(filtered, Blocked):
            observation = self.inner.observe()
            observation = self.filter_observation(observation, pre_state)

            return EnvResponse(
                observation=observation,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "reason": filtered.reason
                }
            )
        
        response = self.inner.step(filtered)
        post_state = self.inner.get_env_state()

        response = self.modify_transition(filtered, response, post_state)
        response.observation = self.filter_observation(response.observation, post_state)

        return response

    def observe(self) -> Observation:
        observation = self.inner.observe()
        env_state = self.inner.get_env_state()
        return self.filter_observation(observation, env_state)

    def reset(self, *args, **kwargs) -> Observation:
        observation = self.inner.reset(*args, **kwargs)
        env_state = self.inner.get_env_state()
        return self.filter_observation(observation, env_state)

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
