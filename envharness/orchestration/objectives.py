from abc import ABC, abstractmethod
from envharness.core.types import ValidationComparison, ObjectiveResult

class Objective(ABC):

    @abstractmethod
    def evaluate(self, comparison: ValidationComparison) -> ObjectiveResult:
        ...

class MinSuccessRateObjective(Objective):

    def __init__(self, min_success_rate: float):
        if not 0.0 <= min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between 0 and 1")
        self.min_success_rate = min_success_rate

    def evaluate(self, comparison):
        candidate_success_rate = comparison.candidate.success_rate
        satisfied = candidate_success_rate >= self.min_success_rate
        return ObjectiveResult(
            satisfied=satisfied,
            score=candidate_success_rate,
            diagnostic=(
                f"candidate success_rate={candidate_success_rate:.2f}, "
                f"required>={self.min_success_rate:.2f}"
            ),
            suggestion=(
                ""
                if satisfied
                else "Increase candidate success rate."
            ),
        )

class ImproveOverBaselineObjective(Objective):

    def __init__(self, min_delta: float = 0.0):
        self.min_delta = min_delta

    def evaluate(self, comparison: ValidationComparison):
        delta = comparison.delta_success_rate
        satisfied = delta >= self.min_delta
        return ObjectiveResult(
            satisfied=satisfied,
            score=delta,
            diagnostic=(
                f"delta success_rate={delta:.2f}, "
                f"required>={self.min_delta:.2f}"
            ),
            suggestion=(
                ""
                if satisfied
                else "Increase candidate success rate."
            ),
        )

class TargetSuccessBandObjective(Objective):

    def __init__(self, low: float = 0.3, high: float = 0.7):
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"low and high must be between {low} and {high}")
        self.low = low
        self.high = high

    def evaluate(self, comparison):
        candidate_success_rate = comparison.candidate.success_rate
        satisfied = self.low <= candidate_success_rate <= self.high
        if candidate_success_rate < self.low:
            diagnostic = "too hard"
            suggestion = "Decrease difficulty" 
        elif candidate_success_rate > self.high:
            diagnostic = "too easy"
            suggestion = "Increase difficulty"
        else:
            diagnostic = "in target band"
            suggestion = ""

        c, h = (self.low + self.high) / 2, (self.high - self.low) / 2
        score = max(0, 1 - abs(candidate_success_rate - c) / h)

        return ObjectiveResult(
            satisfied=satisfied,
            score=score,
            diagnostic=(
                f"candidate successs rate={candidate_success_rate:.2f}, "
                f"target_band=[{self.low:.2f}, {self.high:.2f}], "
                f"{diagnostic}"
            ),
            suggestion=suggestion
        )

