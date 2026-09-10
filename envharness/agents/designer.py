from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from envharness.core.types import Trace, Candidate, Diagnosis, DesignProposal, ValidationBatch, ValidationComparison, ValidationAttempt, DecideResult, Decision, BaselineSnapshot, ObjectiveSignal

@dataclass
class DesignerContext:
    history_traces: list[Trace] = field(default_factory=list)
    task_id: int | str = 0
    task_description: str = ""
    baseline: BaselineSnapshot | None = None
    objective_signal: ObjectiveSignal| None = None


class Designer(ABC):

    def reset(self):
        pass

    @abstractmethod
    def propose(self, ctx: DesignerContext) -> DesignProposal:
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

    def propose(
        self,
        ctx: DesignerContext,
    ) -> DesignProposal:
        diagnosis = Diagnosis(
            summary="scripted proposal"
        )

        candidate = self.write(diagnosis)

        return DesignProposal(
            diagnosis=diagnosis,
            candidate=candidate,
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
