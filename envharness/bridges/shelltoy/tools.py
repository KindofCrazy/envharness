from envharness.core.tool import Tool

class Do(Tool):
    name = "do"
    description = (
        "Send one command to the runtime. "
        "Supported commands: add N, mul N, get, stop"
    )

    @classmethod
    def invoke(cls, env_state, text: str):
        raise NotImplementedError(
            "ShellToyBridge dispatches commands "
            "directly to its runtime"
        )
