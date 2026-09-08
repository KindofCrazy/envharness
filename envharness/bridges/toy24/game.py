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

EPS = 1e-6

def combine(state: Toy24State, i: int, j: int, op: str):
    if state.stopped:
        return {"success": False}

    n = len(state.current_numbers)
    if not (0 <= i < n and 0 <= j < n) or i == j:
        return {
            "success": False
        }

    a, b = state.current_numbers[i], state.current_numbers[j]
    if op == 'add':
        res = a + b
    elif op == 'sub':
        res = a - b
    elif op == 'mul':
        res = a * b
    elif op == 'div':
        if abs(b) < EPS:
            return {"success": False}
        res = a / b
    else:
        return {
            "success": False
        }

    state.current_numbers = [
        n 
        for (index, n) in enumerate(state.current_numbers)
        if index not in (i, j)
    ] + [res]

    state.history.append(f"{a} {op} {b} = {res}")
    return {
        "success": True,
        "res": res
    }

def reset_numbers(state: Toy24State):
    state.current_numbers = [float(n) for n in state.initial_numbers]
    state.history.append('reset')


def stop(state: Toy24State):
    state.stopped = True
    state.success = any(abs(x - state.target) < EPS for x in state.current_numbers)
