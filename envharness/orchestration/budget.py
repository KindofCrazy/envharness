from abc import ABC, abstractmethod
from envharness.core.types import Decision, ObjectiveSignal

class BudgetPolicy(ABC):

    @abstractmethod
    def should_stop(self, attempts: int, last_decision: Decision, objective_signal: ObjectiveSignal) -> bool:
        ...

class FixedBudget(BudgetPolicy):

    def __init__(self, max_attempts: int) :
        if max_attempts < 1:
            raise ValueError("max attempts must be >= 1")
        self.max_attempts = max_attempts

    def should_stop(self, attempts, last_decision, objective_signal):
        return attempts >= self.max_attempts

class CappedAdaptive(BudgetPolicy):

    def __init__(self, max_attempts: int) :
        if max_attempts < 1:
            raise ValueError("max attempts must be >= 1")
        self.max_attempts = max_attempts

    def should_stop(self, attempts, last_decision, objective_signal):
        if last_decision == Decision.ACCEPT:
            return True
        return attempts >= self.max_attempts

class ObjectDriven(BudgetPolicy):

    def __init__(self, score_threshold: float = 0.8, max_attempts: int = 20):
        if max_attempts < 1:
            raise ValueError("max attempts must be >= 1")
        self.score_threshold = score_threshold
        self.max_attempts = max_attempts

    def should_stop(self, attempts, last_decision, objective_signal):
        if attempts >= self.max_attempts:
            return True
        if last_decision == Decision.ACCEPT:
            return True
        if objective_signal is not None and objective_signal.score >= self.score_threshold:
            return True

        return False
