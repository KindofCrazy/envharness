from envharness.core.envharness import EnvHarness
from envharness.core.types import Action, Observation, EnvResponse

class Rules(EnvHarness):
    def __init__(self, inner):
        super().__init__(inner)

    def filter_action(self, action: Action) -> Action:
        return action

    def modify_transition(self, action: Action, response: EnvResponse) -> EnvResponse:
        return response

    def filter_observation(self, observation: Observation) -> Observation:
        return observation

    def step(self, action: Action) -> EnvResponse:
        action = self.filter_action(action)
        response = super().step(action)
        response = self.modify_transition(action, response)
        response.observation = self.filter_observation(response.observation)

        return response

    def observe(self) -> Observation:
        observation = super().observe()
        return self.filter_observation(observation)

    def reset(self, *args, **kwargs) -> Observation:
        observation = super().reset(*args, **kwargs)
        return self.filter_observation(observation)
    