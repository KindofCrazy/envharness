from abc import ABC, abstractmethod
from envharness.core.types import ObjectiveSignal, Trace, TraceKind

class MutationObjective(ABC):

    @abstractmethod
    def evaluate(self, recent_traces: list[Trace]) -> ObjectiveSignal:
        ...

class DifficultyZone(MutationObjective):

    def __init__(self, low: float = 0.3, high: float = 0.7, window: int = 10):
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("low and high must satisfy 0 <= low < high <= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.low = low
        self.high = high
        self.window = window

    def evaluate(self, recent_traces):
        accepted = [trace for trace in recent_traces if trace.kind == TraceKind.ACCEPTED]
        accepted = accepted[-self.window:]
        if not accepted:
            return ObjectiveSignal(
                score=0.0,
                diagnostic="no accepted traces yet",
                suggestion="Collect accepted traces",
            )

        success_rate = sum(trace.success for trace in accepted) / len(accepted)
        center, half_width = (self.low + self.high) / 2, (self.high - self.low) / 2
        score = max(0.0, 1.0 - abs(success_rate - center) / half_width)
        if success_rate < self.low:
            diagnostic = "accepted tasks are too hard"
            suggestion = "Decrease difficulty"
        elif success_rate > self.high:
            diagnostic = "accepted tasks are too easy"
            suggestion = "Increase difficulty"
        else:
            diagnostic = "accepted tasks are in target band"
            suggestion = "Maintain difficulty"

        return ObjectiveSignal(
            score=score,
            diagnostic=(
                f"accepted success_rate="
                f"{success_rate:.2f}, "
                f"target_band="
                f"[{self.low:.2f}, {self.high:.2f}], "
                f"{diagnostic}"
            ),
            suggestion=suggestion
        )
