from envharness.core.envharness import EnvHarness
from envharness.core.types import Action, Blocked, Observation, EnvResponse, EnvResetResponse
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
            fresh = self.inner.observe()

            fresh = self.filter_observation(
                fresh,
                pre_state,
            )

            observation = Observation(
                text=(
                    f"[blocked] {filtered.reason}\n\n"
                    f"{fresh.text}"
                ),
                data={
                    **fresh.data,
                    "blocked": True,
                    "blocked_reason": filtered.reason,
                },
            )

            return EnvResponse(
                observation=observation,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "blocked_reason": filtered.reason,
                },
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

    def reset(self, *args, **kwargs) -> EnvResetResponse:
        reset_response = self.inner.reset(
            *args,
            **kwargs,
        )

        env_state = self.inner.get_env_state()

        observation = self.filter_observation(
            self.inner.observe(),
            env_state,
        )

        return EnvResetResponse(
            observation=observation,
            info=dict(reset_response.info),
        )
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
