from typing import Any
from dataclasses import dataclass, field
from envharness.core.types import DesignProposal, ValidationBatch, DecideResult, BaselineSnapshot, Candidate, Trace, Decision, TraceKind
from envharness.orchestration.runner import run_episode
from envharness.orchestration.builder import build_env_stack
from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.agents.designer import Designer, DesignerContext
from envharness.orchestration.budget import BudgetPolicy
from envharness.orchestration.baseline import summarize_baseline
from envharness.orchestration.objectives import MutationObjective

@dataclass
class OrchestratorAttempt:
    proposal: DesignProposal
    validation: ValidationBatch
    decision: DecideResult

@dataclass
class OrchestratorResult:
    baseline: BaselineSnapshot
    baseline_validation: ValidationBatch
    attempts: list[OrchestratorAttempt] = field(default_factory=list)
    accepted_candidate: Candidate | None = None

    @property
    def traces(self) -> list[Trace]:
        result = list(self.baseline_validation.traces)
        for attempt in self.attempts:
            result.extend(attempt.validation.traces)
        return result

@dataclass
class TaskSpec:
    task_id: int | str
    task_description: str = ""
    reset_args: tuple[Any, ...] = field(default_factory=tuple)
    reset_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class OrchestratorRunResult:
    task_results: list[OrchestratorResult] = field(default_factory=list)
    history_traces: list[Trace] = field(default_factory=list)

def _evaluate_env_k(
    env,
    policy, 
    k, 
    *reset_args, 
    trace_kind: TraceKind = TraceKind.EXPLORATION,
    task_id: int | str | None = None,
    attempt_idx: int | None = None,
    **reset_kwargs
) -> ValidationBatch:
    if k <= 0:
        raise ValueError("validation_rollouts must be positive")

    traces = []
    for rollout_idx in range(k):
        traces.append(run_episode(env, policy, *reset_args, trace_kind=trace_kind, task_id=task_id, attempt_idx=attempt_idx, rollout_idx=rollout_idx, **reset_kwargs))

    return ValidationBatch(
        traces=list(traces)
    )

def _evaluate_candidate_k(
    base_env, 
    candidate, 
    policy, 
    k, 
    *reset_args, 
    trace_kind: TraceKind = TraceKind.EXPLORATION,
    task_id: int | str | None = None,
    attempt_idx: int | None = None,
    **reset_kwargs
) -> ValidationBatch:
    candidate_env = build_env_stack(base_env, candidate)
    return _evaluate_env_k(candidate_env, policy, k, *reset_args, trace_kind=trace_kind, task_id=task_id, attempt_idx=attempt_idx, **reset_kwargs)

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
    objective: MutationObjective | None = None,
    **reset_kwargs,
) -> OrchestratorResult:
    history = list(history_traces or [])
    objective_signal = (
        objective.evaluate(history)
        if objective is not None
        else None
    )

    baseline_batch = _evaluate_env_k(base_env, policy, validation_rollouts, *reset_args, trace_kind=TraceKind.BASELINE, task_id=task_id, **reset_kwargs)
    baseline = summarize_baseline(baseline_batch)

    ctx = DesignerContext(
        history_traces=history,
        task_id=task_id,
        task_description=task_description,
        baseline=baseline,
        objective_signal=objective_signal,
    )
    proposal = designer.propose(ctx)

    attempts = []
    accepted_candidate = None
    while True:
        attempt_idx = len(attempts)
        validation = _evaluate_candidate_k(base_env, proposal.candidate, policy, validation_rollouts, *reset_args, trace_kind=TraceKind.EXPLORATION, task_id=task_id, attempt_idx=attempt_idx, **reset_kwargs)
        decision = designer.decide(proposal.candidate, validation, ctx)

        attempts.append(
            OrchestratorAttempt(
                proposal=proposal,
                validation=validation,
                decision=decision,
            )
        )
        ctx.history_traces.extend(validation.traces)

        if decision.decision == Decision.ACCEPT:
            for trace in attempts[-1].validation.traces:
                trace.kind = TraceKind.ACCEPTED
            accepted_candidate = proposal.candidate
            break

        if budget.should_stop(len(attempts), decision.decision, objective_signal):
            break

        if decision.decision == Decision.REFINE:
            proposal = designer.refine(proposal.candidate, validation, ctx)
        elif decision.decision == Decision.REJECT:
            proposal = designer.propose(ctx)

    return OrchestratorResult(
        baseline=baseline,
        baseline_validation=baseline_batch,
        attempts=attempts,
        accepted_candidate=accepted_candidate
    )

def run_orchestrator(
    base_env: ActionableEnv,
    policy: Policy,
    designer: Designer,
    budget: BudgetPolicy,
    tasks: list[TaskSpec],
    *,
    validation_rollouts: int = 5,
    objective: MutationObjective | None = None,
) -> OrchestratorRunResult:
    designer.reset()

    history: list[Trace] = []
    task_results = []

    for task in tasks:
        result = run_orchestrator_task(
            base_env=base_env,
            policy=policy,
            designer=designer,
            budget=budget,
            *task.reset_args,
            task_id=task.task_id,
            task_description=task.task_description,
            history_traces=history,
            validation_rollouts=validation_rollouts,
            objective=objective,
            **task.reset_kwargs,
        )

        task_results.append(result)
        history.extend(result.traces)

    return OrchestratorRunResult(
        task_results=task_results,
        history_traces=history,
    )