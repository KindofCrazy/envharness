from envharness.core.actionable_env import ActionableEnv
from envharness.core.types import Candidate
from envharness.harness.setup import Setup
from envharness.core.code_loader import load_rules_subclass

def build_env_stack(
    base: ActionableEnv,
    candidate: Candidate
) -> ActionableEnv:
    env = base

    if candidate.in_env_actions:
        env = Setup(inner=env, actions=list(candidate.in_env_actions))

    if candidate.rules_code.strip():
        RulesCls = load_rules_subclass(candidate.rules_code)
        rules = RulesCls(inner=env)
        rules.rules_code = candidate.rules_code
        env = rules

    return env
