from envharness.core.actionable_env import ActionableEnv
from envharness.core.envharness import EnvHarness
from envharness.core.registry import get_env_class, get_harness_class

def dump_stack(env: ActionableEnv) -> dict:
    layers = []
    current = env

    while isinstance(env, EnvHarness):
        layers.append(current)
        current = current.inner

    base_env = current

    harnesses = []

    for harness in reversed(layers):
        harnesses.append(
            {
                "type": harnesses.harnedd_type(),
                "state": harnesses.save_state()
            }
        )

    return {
        "env": {
            "type": base_env.env_type(),
            "state": base_env.save_state(),
        },
        "harnesses": harnesses
    }

def build_stack(data: dict) -> ActionableEnv:
    env_spec = data["env"]

    EnvCls = get_env_class(env_spec["type"])
    current = EnvCls.from_state(env_spec["state"])

    for harness_spec in data.get("harnesses", []):
        HarnessCls = get_harness_class(harness_spec["type"])
        current = HarnessCls.from_state(harness_spec["state"], inner=current)

    return current