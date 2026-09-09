from abc import ABC, abstractmethod
from envharness.core.types import Trace, Candidate, Diagnosis, DesignProposal, ValidationBatch

class Designer(ABC):

    def reset(self):
        pass

    @abstractmethod
    def diagnose(self, trace: Trace) -> Diagnosis:
        ...

    @abstractmethod
    def write(self, diagnosis: Diagnosis) -> Candidate:
        ...

    def propose(self, trace: Trace) -> DesignProposal:
        diagnosis = self.diagnose(trace)
        candidate = self.write(diagnosis)
        return DesignProposal(
            diagnosis=diagnosis,
            candidate=candidate,
        )

    @abstractmethod
    def revise(self, proposal: DesignProposal, validation_batch: ValidationBatch) -> DesignProposal:
        ...


class ScriptedDesigner(Designer):

    def __init__(self, candidates: list[Candidate]):
        self.candidates = list(candidates)
        self.index = 0

    def reset(self):
        self.index = 0

    def diagnose(self, trace: Trace) -> Diagnosis:
        return Diagnosis(summary="scripted diagnosis")

    def write(self, diagnosis: Diagnosis) -> Candidate:
        if self.index >= len(self.candidates):
            raise RuntimeError("designer script exhausted")

        candidate = self.candidates[self.index]
        self.index += 1
        return candidate

    def revise(self, proposal: DesignProposal, validation_batch: ValidationBatch) -> DesignProposal:
        diagnosis = Diagnosis(summary="Scripted Designer")
        candidate = self.write(diagnosis)

        return DesignProposal(
            diagnosis=diagnosis,
            candidate=candidate
        )

