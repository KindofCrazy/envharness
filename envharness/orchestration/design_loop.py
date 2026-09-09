from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.agents.designer import Designer
from envharness.core.types import Trace, Candidate, DesignProposal, ValidationBatch
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
    *reset_args,
    revision_budget: int = 1,
    validation_rollouts: int = 5,
    min_success_rate: float = 0.5,
    **reset_kwargs,
) -> tuple[list[tuple[Trace, list[tuple[DesignProposal, ValidationBatch]], bool]], ActionableEnv]:
    designer.reset()

    current_env = base_env
    history = []

    while num_iterations > 0:
        num_iterations -= 1
        proposal_trace = run_episode(current_env, policy, *reset_args, **reset_kwargs)
        proposal = designer.propose(proposal_trace)

        attempts = validate_with_revisions(base_env, proposal, policy, designer, *reset_args, revision_budget=revision_budget, num_rollouts=validation_rollouts, min_success_rate=min_success_rate, **reset_kwargs)
        final_proposal, final_validation_batch = attempts[-1]
        accepted = validation_passes(final_validation_batch, min_success_rate)
        current_env, accepted = select_next_env(base_env, current_env, final_proposal.candidate, accepted)
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

def validate_candidate_k(
    base_env: ActionableEnv,
    candidate: Candidate,
    policy: Policy,
    num_rollouts: int,
    *reset_args,
    **reset_kwargs,
) -> ValidationBatch:
    if (num_rollouts <= 0):
        raise ValueError("num_rollouts must be positive")

    candidate_env = build_env_stack(base_env, candidate)

    traces = []
    for _ in range(num_rollouts):
        trace = run_episode(candidate_env, policy, *reset_args, **reset_kwargs)
        traces.append(trace)

    return ValidationBatch(
        traces=list(traces)
    )

def validation_passes(
    batch: ValidationBatch,
    min_success_rate: float
) -> bool:
    if not 0 <= min_success_rate <= 1.0:
        raise ValueError("min_success_rate must be between 0 and 1")
    return batch.success_rate >= min_success_rate

def select_next_env(
    base_env: ActionableEnv,
    current_env: ActionableEnv,
    candidate: Candidate,
    accepted: bool
) -> tuple[ActionableEnv, bool]:
    if accepted:
        return build_env_stack(base_env, candidate), True
    return current_env, False

def validate_with_revisions(
    base_env: ActionableEnv,
    proposal: DesignProposal,
    policy: Policy,
    designer: Designer,
    *reset_args,
    max_revisions: int,
    num_rollouts: int,
    min_success_rate: float,
    **reset_kwargs,
) -> list[tuple[DesignProposal, ValidationBatch]]:
    attempts = []

    validation_batch = validate_candidate_k(base_env, proposal.candidate, policy, num_rollouts, *reset_args, **reset_kwargs)
    attempts.append((proposal, validation_batch))

    if validation_passes(validation_batch, min_success_rate):
        return attempts

    while max_revisions > 0:
        max_revisions -= 1
        proposal = designer.revise(proposal, validation_batch)
        validation_batch = validate_candidate_k(base_env, proposal.candidate, policy, num_rollouts, *reset_args, **reset_kwargs)
        attempts.append((proposal, validation_batch))

        if validation_passes(validation_batch, min_success_rate):
            break

    return attempts
