from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from envharness.core.types import Trace, Candidate, Diagnosis, DesignProposal, ValidationBatch, DecideResult, BaselineSnapshot, ObjectiveSignal

@dataclass
class DesignerContext:
    history_traces: list[Trace] = field(default_factory=list)
    tool_schema: list[dict] = field(default_factory=list)
    env_state_schema: str = ""

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

    def _next_candidate(self) -> Candidate:
        if self.index >= len(self.candidates):
            raise RuntimeError("designer candidate script exhausted")
        candidate = self.candidates[self.index]
        self.index += 1
        return candidate

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

        return DesignProposal(
            diagnosis=diagnosis,
            candidate=self._next_candidate(),
        )

    def decide(self, candidate: Candidate, validation: ValidationBatch, ctx: DesignerContext) -> DecideResult:
        if self.decision_index >= len(self.decisions):
            raise RuntimeError("designer decision script exhausted")

        result = self.decisions[self.decision_index]
        self.decision_index += 1
        return result

    def refine(self, candidate: Candidate, validation: ValidationBatch, ctx: DesignerContext) -> DesignProposal:
        diagnosis = Diagnosis(summary="Scripted Designer")

        return DesignProposal(
            diagnosis=diagnosis,
            candidate=self._next_candidate(),
        )
