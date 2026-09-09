from envharness.core.envharness import EnvHarness
from envharness.core.types import Action, Observation, EnvResponse

class Rules(EnvHarness):
    def __init__(self, inner):
        super().__init__(inner)

    def filter_action(self, action: Action) -> Action:
        return action

    def filter_transition(self, action: Action, response: EnvResponse) -> EnvResponse:
        return response

    def filter_observation(self, observation: Observation) -> Observation:
        return observation

    def step(self, action):
        action = self.filter_action(action)
        response = super().step()
        response = self.filter_transition(action, response)
        response.observation = self.filter_observation(response.observation)

        return response

    def observe(self):
        observation = super().observe()
        return self.filter_observation(observation)