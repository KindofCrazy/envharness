from dataclasses import dataclass

@dataclass
class Toy24State:
    target: int
    initial_numbers: list[int]
    current_numbers: list[float]
    operation_history: list[str]
    stopped: bool
    success: bool
    step_count: int