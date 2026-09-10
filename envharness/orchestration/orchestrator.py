from dataclasses import dataclass, field
from envharness.core.types import DesignProposal, ValidationBatch, DecideResult, BaselineSnapshot, Candidate, Trace, Decision
from envharness.orchestration.runner import run_episode
from envharness.orchestration.builder import build_env_stack
from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.agents.designer import Designer, DesignerContext
from envharness.orchestration.budget import BudgetPolicy
from envharness.orchestration.baseline import summarize_baseline

@dataclass
class OrchestratorAttempt:
    proposal: DesignProposal
    validation: ValidationBatch
    decision: DecideResult

@dataclass
class OrchestratorResult:
    baseline: BaselineSnapshot
    attempts: list[OrchestratorAttempt] = field(default_factory=list)
    accepted_candidate: Candidate | None = None

def _evaluate_env_k(env, policy, k, *reset_args, **reset_kwargs) -> ValidationBatch:
    if k <= 0:
        raise ValueError("validation_rollouts must be positive")

    traces = []
    for _ in range(k):
        traces.append(run_episode(env, policy, *reset_args, **reset_kwargs))

    return ValidationBatch(
        traces=list(traces)
    )

def _evaluate_candidate_k(base_env, candidate, policy, k, *reset_args, **reset_kwargs) -> ValidationBatch:
    candidate_env = build_env_stack(base_env, candidate)
    return _evaluate_env_k(candidate_env, policy, k, *reset_args, **reset_kwargs)

def run_orchestrator_task(
    base_env: ActionableEnv,
    policy: Policy,
    designer: Designer,
    budget: BudgetPolicy,
    *reset_args,
    task_id: int | str = 0,
    task_description: str = "",
    history_traces: list[Trace] | None = None,
    validation_rollouts: int = 5,
    **reset_kwargs,
) -> OrchestratorResult:
    baseline_batch = _evaluate_env_k(base_env, policy, validation_rollouts, *reset_args, **reset_kwargs)
    baseline = summarize_baseline(baseline_batch)

    ctx = DesignerContext(
        history_traces=list(history_traces or []),
        task_id=task_id,
        task_description=task_description,
        baseline=baseline
    )
    proposal = designer.propose(ctx)

    attempts = []
    accepted_candidate = None
    while True:
        validation = _evaluate_candidate_k(base_env, proposal.candidate, policy, validation_rollouts, *reset_args, **reset_kwargs)
        decision = designer.decide(proposal.candidate, validation, ctx)

        attempts.append(
            OrchestratorAttempt(
                proposal=proposal,
                validation=validation,
                decision=decision
            )
        )
        ctx.history_traces.extend(validation.traces)

        if decision.decision == Decision.ACCEPT:
            accepted_candidate = proposal.candidate
            break

        if budget.should_stop(len(attempts), decision.decision, None):
            break

        if decision.decision == Decision.REFINE:
            proposal = designer.refine(proposal.candidate, validation, ctx)
        elif decision.decision == Decision.REJECT:
            proposal = designer.propose(ctx)

    return OrchestratorResult(
        baseline=baseline,
        attempts=attempts,
        accepted_candidate=accepted_candidate
    )