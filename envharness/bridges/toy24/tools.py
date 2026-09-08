from envharness.core.tool import Tool
from envharness.bridges.toy24.game import combine, reset_numbers, stop

class Combine(Tool):
    name = "combine"
    description = "Combine two numbers using an arithmetic operation."

    @classmethod
    def invoke(cls, env_state, i: int, j: int, op: str):
        return combine(env_state, i, j, op)

class Reset(Tool):
    name = "reset"
    description = "Reset to the initial numbers"

    @classmethod
    def invoke(cls, env_state):
        reset_numbers(env_state)
        return {
            "ok": True,
            "numbers": list(env_state.current_numbers),
        }

class Stop(Tool):
    name = "stop"
    description = "Stop"

    @classmethod
    def invoke(cls, env_state):
        stop(env_state)
        return {
            "ok": True,
            "numbers": list(env_state.current_numbers),
        }