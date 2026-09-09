from abc import ABC, abstractmethod
from envharness.core.types import ValidationComparison

class Objective(ABC):

    @abstractmethod
    def satisfied(self, comparison: ValidationComparison) -> bool:
        ...

class MinSuccessRateObjective(Objective):

    def __init__(self, min_success_rate: float):
        if not 0.0 <= min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between 0 and 1")
        self.min_success_rate = min_success_rate

    def satisfied(self, comparison):
        return comparison.candidate.success_rate >= self.min_success_rate

class ImproveOverBaselineObjective(Objective):

    def __init__(self, min_delta: float = 0.0):
        self.min_delta = min_delta

    def satisfied(self, comparison):
        return comparison.delta_success_rate >= self.min_delta