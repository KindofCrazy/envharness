from envharness.bridges.toy24.game import Toy24State

def test_toy_state_initialization():
    state = Toy24State (
        target=24,
        initial_numbers=[3, 3, 8, 8],
        current_numbers=[3.0, 3.0, 8.0, 8.0],
        history=[],
        step_count=0,
        stopped=False,
        success=False
    )

    assert state.target == 24
    assert state.initial_numbers == [3, 3, 8, 8]
    assert state.step_count == 0