from abc import ABC, abstractmethod
from envharness.core.types import Trace, Candidate

class Designer(ABC):

    def reset(self):
        pass

    @abstractmethod
    def propose(self, trace: Trace) -> Candidate:
        ...