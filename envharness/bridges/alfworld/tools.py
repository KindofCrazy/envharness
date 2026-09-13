from envharness.core.tool import Tool


class Do(Tool):
    name = "do"

    description = (
        "Issue one natural-language command to the ALFWorld environment. "
        "Prefer a command listed in the current observation's "
        "admissible_commands."
    )

    @classmethod
    def invoke(
        cls,
        env_state,
        text: str,
    ):
        raise NotImplementedError(
            "Do.invoke() is not used for ALFWorld. "
            "AlfworldEnv.step() dispatches commands directly "
            "to the underlying TextWorld runtime."
        )