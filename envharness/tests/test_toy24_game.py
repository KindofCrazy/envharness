from toy24.game import Toy24State

state = Toy24State (
    target=24,
    initial_numbers=[3, 3, 8, 8],
    current_numbers=[3, 3, 8, 8],
    step_count=0,
    stopped=True,
    success=True,
)

assert state.target == 24
assert state.initial_numbers == [3, 3, 8, 8]
assert state.step_count == 0