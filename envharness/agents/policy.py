from abc import ABC, abstractmethod
from envharness.core.types import Action, Observation

class Policy(ABC):

    def reset(self):
        pass

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        ...

class ScriptedPolicy(Policy):
    def __init__(self, actions: list[Action]):
        self.actions = list(actions)
        self.index = 0

    def reset(self):
        self.index = 0

    def act(self, observation: Observation) -> Action:
        if self.index >= len(self.actions):
            raise RuntimeError("script exhausted")
        action = self.actions[self.index]
        self.index += 1
        return action

