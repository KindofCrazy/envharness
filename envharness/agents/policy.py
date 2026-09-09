from abc import ABC, abstractmethod
from envharness.core.types import Action, Observation

class Policy(ABC):

    def reset(self):
        pass

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        ...