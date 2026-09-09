from envharness.harness.rules import Rules
from envharness.core.types import Action, Blocked, Observation, EnvResponse

class RulesCodeError(Exception):
    pass

def load_rules_subclass(code: str) -> type:
    if not code.strip():
        return Rules

    namespace = {
        "Action": Action,
        "Blocked": Blocked,
        "Observation": Observation,
        "EnvResponse": EnvResponse,
        "Rules": Rules
    }
    try:
        compiled = compile(code, "<rules>", "exec")
    except SyntaxError as e:
        raise RulesCodeError(
            f"SyntaxError: {e}"
        ) from e

    try:
        exec(compiled, namespace)
    except Exception as e:
        raise RulesCodeError(
            f"Execution failed: {e}"
        ) from e

    cls = namespace.get("_Rules")
    if cls is None:
        raise RulesCodeError(
            "Code must define _Rules"
        )
    if not isinstance(cls, type):
        raise RulesCodeError(
            "_Rules must be a class"
        )
    if not issubclass(cls, Rules):
        raise RulesCodeError(
            "_Rules must subclass Rules"
        )

    return cls

