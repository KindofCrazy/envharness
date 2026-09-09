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
) -> tuple[Trace, DesignProposal, ActionableEnv]:
    trace = run_episode(current_env, policy, *reset_args, **reset_kwargs)
    proposal = designer.propose(trace)
    candidate = proposal.candidate
    new_env = build_env_stack(base_env, candidate)

    return trace, proposal, new_env

def run_design_loop(
    base_env: ActionableEnv,
    policy: Policy,
    designer: Designer,
    num_iterations: int,
    revision_budget: int = 1,
    *reset_args,
    **reset_kwargs,
) -> tuple[list[tuple[Trace, list[tuple[DesignProposal, Trace]], bool]], ActionableEnv]:
    designer.reset()

    current_env = base_env
    history = []

    while num_iterations > 0:
        num_iterations -= 1
        proposal_trace = run_episode(current_env, policy, *reset_args, **reset_kwargs)
        proposal = designer.propose(proposal_trace)

        attempts = validate_with_revisions(base_env, proposal, policy, designer, revision_budget, *reset_args, **reset_kwargs)
        final_proposal, final_validation_trace = attempts[-1]
        current_env, accepted = select_next_env(base_env, current_env, final_proposal.candidate, final_validation_trace)

        history.append((proposal_trace, attempts, accepted))

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

def validate_with_one_revision(
    base_env: ActionableEnv,
    proposal: DesignProposal,
    policy: Policy,
    designer: Designer,
    *reset_args,
    **reset_kwargs,
) -> list[tuple[DesignProposal, Trace]]:
    attempts = []

    validation_trace = validate_candidate(base_env, proposal.candidate, policy, *reset_args, **reset_kwargs)
    attempts.append((proposal, validation_trace))

    if validation_trace.success:
        return attempts

    proposal = designer.revise(proposal, validation_trace)
    validation_trace = validate_candidate(base_env, proposal.candidate, policy, *reset_args, **reset_kwargs)
    attempts.append((proposal, validation_trace))

    return attempts

def validate_with_revisions(
    base_env: ActionableEnv,
    proposal: DesignProposal,
    policy: Policy,
    designer: Designer,
    max_revisions: int,
    *reset_args,
    **reset_kwargs,
) -> list[tuple[DesignProposal, Trace]]:
    attempts = []

    validation_trace = validate_candidate(base_env, proposal.candidate, policy, *reset_args, **reset_kwargs)
    attempts.append((proposal, validation_trace))

    if validation_trace.success:
        return attempts

    while max_revisions > 0:
        max_revisions -= 1
        proposal = designer.revise(proposal, validation_trace)
        validation_trace = validate_candidate(base_env, proposal.candidate, policy, *reset_args, **reset_kwargs)
        attempts.append((proposal, validation_trace))

        if validation_trace.success:
            break

    return attempts
