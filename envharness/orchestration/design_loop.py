from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.agents.designer import Designer
from envharness.core.types import Trace, Candidate
from envharness.orchestration.builder import build_env_stack
from envharness.orchestration.runner import run_episode

def run_design_step(
    base_env: ActionableEnv,
    current_env: ActionableEnv,
    policy: Policy,
    designer: Designer,
    *reset_args,
    **reset_kwargs,
) -> tuple[Trace, Candidate, ActionableEnv]:
    trace = run_episode(current_env, policy, *reset_args, **reset_kwargs)
    candidate = designer.propose(trace)
    new_env = build_env_stack(base_env, candidate)

    return trace, candidate, new_env