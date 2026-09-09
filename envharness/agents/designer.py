from abc import ABC, abstractmethod
from envharness.core.types import Trace, Candidate

class Designer(ABC):

    def reset(self):
        pass

    @abstractmethod
    def propose(self, trace: Trace) -> Candidate:
        ...

class ScriptedDesigner(Designer):

    def __init__(self, candidates: list[Candidate]):
        self.candidate = list(candidates)
        self.index = 0

    def reset(self):
        self.index = 0

    def propose(self, trace: Trace) -> Candidate:
        if self.index >= len(self.candidate):
            raise RuntimeError("designer script exhausted")

        candidate = self.candidate[self.index]
        self.index += 1
        return candidate