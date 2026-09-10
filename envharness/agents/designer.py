from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from envharness.core.types import Trace, Candidate, Diagnosis, DesignProposal, ValidationBatch, ValidationComparison, ValidationAttempt, DecideResult, Decision, BaselineSnapshot

@dataclass
class DesignerContext:
    history_traces: list[Trace] = field(default_factory=list)
    task_id: int | str = 0
    task_description: str = ""
    baseline: BaselineSnapshot | None = None


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
    def revise(self, attempt: ValidationAttempt) -> DesignProposal:
        ...

    @abstractmethod
    def decide(self, candidate: Candidate, validation: ValidationBatch, ctx: DesignerContext) -> DecideResult:
        ...

    @abstractmethod
    def refine(self, candidate: Candidate, validation: ValidationBatch, ctx: DesignerContext) -> DesignProposal:
        ...


class ScriptedDesigner(Designer):

    def __init__(self, candidates: list[Candidate], decisions: list[DecideResult] | None = None):
        self.candidates = list(candidates)
        self.decisions = list(decisions or [])
        self.index = 0
        self.decision_index = 0

    def reset(self):
        self.index = 0
        self.decision_index = 0

    def diagnose(self, trace: Trace) -> Diagnosis:
        return Diagnosis(summary="scripted diagnosis")

    def write(self, diagnosis: Diagnosis) -> Candidate:
        if self.index >= len(self.candidates):
            raise RuntimeError("designer script exhausted")

        candidate = self.candidates[self.index]
        self.index += 1
        return candidate

    def revise(self, attempt: ValidationAttempt) -> DesignProposal:
        diagnosis = Diagnosis(summary="Scripted Designer")
        candidate = self.write(diagnosis)

        return DesignProposal(
            diagnosis=diagnosis,
            candidate=candidate
        )

    def decide(self, candidate: Candidate, validation: ValidationBatch, ctx: DesignerContext) -> DecideResult:
        if self.decision_index >= len(self.decisions):
            raise RuntimeError("designer decision script exhausted")

        result = self.decisions[self.decision_index]
        self.decision_index += 1
        return result

    def refine(self, candidate: Candidate, validation: ValidationBatch, ctx: DesignerContext) -> DesignProposal:
        diagnosis = Diagnosis(summary="Scripted Designer")
        candidate = self.write(diagnosis)

        return DesignProposal(
            diagnosis=diagnosis,
            candidate=candidate
        )
