from dataclasses import dataclass, field

@dataclass
class Toy24State:
    target: int = 24
    initial_numbers: list[int] = field(default_factory=list)
    current_numbers: list[float] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    stopped: bool = False
    success: bool = False
    step_count: int = 0