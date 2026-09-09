from envharness.harness.rules import Rules
from envharness.core.types import Action, Blocked, Observation, EnvResponse

def load_rules_subscalss(code: str) -> type:
    if code == "":
        return Rules

    namespace = {
        "Action": Action,
        "Blocked": Blocked,
        "Observation": Observation,
        "EnvResponse": EnvResponse
    }

    complie(code, namespace, exec)

class RulesCodeError(Exception):
