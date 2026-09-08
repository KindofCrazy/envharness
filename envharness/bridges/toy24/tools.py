from envharness.core.tool import Tool
from envharness.bridges.toy24.game import combine, reset_numbers, stop

class Combine(Tool):
    name = "combine"

    @classmethod
    def invoke(cls, env_state, i, j , op):
        return combine(env_state, i, j, op)

class Reset(Tool):
    name = "reset"

    @classmethod
    def invoke(cls, env_state):
        return reset_numbers(env_state)

class Stop(Tool):
    name = "stop"

    @classmethod
    def invoke(cls, env_state):
        return stop(env_state)