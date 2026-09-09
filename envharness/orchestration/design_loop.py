from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.agents.designer import Designer
from envharness.core.types import Trace, Candidate, DesignProposal
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

def run_design_loop(
    base_env: ActionableEnv,
    policy: Policy,
    designer: Designer,
    num_iterations: int,
    *reset_args,
    **reset_kwargs,
) -> tuple[list[tuple[Trace, DesignProposal, Trace, bool]], ActionableEnv]:
    designer.reset()

    current_env = base_env
    history = []

    while num_iterations > 0:
        num_iterations -= 1
        proposal_trace = run_episode(current_env, policy, *reset_args, **reset_kwargs)
        proposal = designer.propose(proposal_trace)
        candidate = proposal.candidate
    
        validation_trace = validate_candidate(base_env, candidate, policy, *reset_args, **reset_kwargs)
        current_env, accepted = select_next_env(base_env, current_env, candidate, validation_trace)

        history.append((proposal_trace, proposal, validation_trace, accepted))

    return history, current_env

def validate_candidate(
    base_env: ActionableEnv,
    candidate: Candidate,
    policy: Policy,
    *reset_args,
    **reset_kwargs,
) -> Trace:
    candidate_env = build_env_stack(base_env, candidate)
    validation_trace = run_episode(candidate_env, policy, *reset_args, **reset_kwargs)
    return validation_trace

def select_next_env(
    base_env: ActionableEnv,
    current_env: ActionableEnv,
    candidate: Candidate,
    validation_trace: Trace
) -> tuple[ActionableEnv, bool]:
    if validation_trace.success:
        candidate_env = build_env_stack(base_env, candidate)
        return candidate_env, True
    return current_env, False