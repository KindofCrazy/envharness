from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.agents.designer import Designer
from envharness.core.types import Trace, Candidate, DesignProposal, ValidationBatch, ValidationComparison, ValidationAttempt, DesignIterationResult
from envharness.orchestration.objectives import Objective
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
    objective: Objective,
    validation_rollouts: int = 5,
    **reset_kwargs,
) -> tuple[list[DesignIterationResult], ActionableEnv]:
    designer.reset()
    current_env = base_env
    history = []

    while num_iterations > 0:
        num_iterations -= 1

        proposal_trace = run_episode(current_env, policy, *reset_args, **reset_kwargs)
        proposal = designer.propose(proposal_trace)

        baseline_batch = evaluate_env_k(current_env, policy, validation_rollouts, *reset_args, **reset_kwargs)
        attempts = validate_with_revisions(base_env, proposal, policy, designer, *reset_args, max_revisions=revision_budget, num_rollouts=validation_rollouts, baseline_batch=baseline_batch, objective=objective, **reset_kwargs)
        final_attempt = attempts[-1]

        accepted = final_attempt.objective_result.satisfied

        current_env, accepted = select_next_env(base_env, current_env, final_attempt.proposal.candidate, accepted)
        history.append(
            DesignIterationResult(
                baseline=baseline_batch,
                proposal_trace=proposal_trace,
                attempts=attempts,
                accepted=accepted
            )
        )

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
    candidate_env = build_env_stack(base_env, candidate)
    return evaluate_env_k(candidate_env, policy, num_rollouts, *reset_args, **reset_kwargs)

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
    baseline_batch: ValidationBatch,
    objective: Objective,
    **reset_kwargs,
) -> list[ValidationAttempt]:
    attempts = []

    validation_batch = validate_candidate_k(base_env, proposal.candidate, policy, num_rollouts, *reset_args, **reset_kwargs)
    comparison = ValidationComparison(
        baseline=baseline_batch,
        candidate=validation_batch
    )
    objective_result = objective.evaluate(comparison)
    attempt = ValidationAttempt(
        proposal=proposal,
        validation=validation_batch,
        comparison=comparison,
        objective_result=objective_result
    )
    attempts.append(attempt)
    if objective_result.satisfied:
        return attempts

    while max_revisions > 0:
        max_revisions -= 1
        proposal = designer.revise(attempt)
        validation_batch = validate_candidate_k(base_env, proposal.candidate, policy, num_rollouts, *reset_args, **reset_kwargs)

        comparison = ValidationComparison(
            baseline=baseline_batch,
            candidate=validation_batch
        )
        objective_result = objective.evaluate(comparison)
        attempt = ValidationAttempt(
            proposal=proposal,
            validation=validation_batch,
            comparison=comparison,
            objective_result=objective_result
        )
        attempts.append(attempt)
        if objective_result.satisfied:
            return attempts

    return attempts

def evaluate_env_k(
    env: ActionableEnv,
    policy: Policy,
    num_rollouts: int,
    *reset_args,
    **reset_kwargs,
) -> ValidationBatch:
    if (num_rollouts <= 0):
        raise ValueError("num_rollouts must be positive")

    traces = []
    for _ in range(num_rollouts):
        trace = run_episode(env, policy, *reset_args, **reset_kwargs)
        traces.append(trace)

    return ValidationBatch(
        traces=list(traces)
    )
