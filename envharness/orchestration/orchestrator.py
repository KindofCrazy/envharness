from typing import Any
from dataclasses import dataclass, field, replace
from envharness.core.types import Observation, DesignProposal, ValidationBatch, DecideResult, BaselineSnapshot, Candidate, Trace, Decision, TraceKind
from envharness.infra.utils import import_symbol
from envharness.orchestration.runner import run_episode, run_episode_spec, EpisodeRunner
from envharness.orchestration.builder import build_env_stack, build_episode_env
from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.agents.designer import Designer, DesignerContext
from envharness.orchestration.budget import BudgetPolicy
from envharness.orchestration.baseline import summarize_baseline
from envharness.orchestration.objectives import MutationObjective
from envharness.orchestration.storage import TraceStore
from envharness.orchestration.specs import EnvSpec, PolicySpec, EpisodeSpec


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

def _evaluate_candidate_k(
    env_spec: EnvSpec,
    policy_spec: PolicySpec,
    runner: EpisodeRunner,
    candidate: Candidate,
    k: int,
    *reset_args,
    trace_kind: TraceKind = TraceKind.EXPLORATION,
    task_id: int | str | None = None,
    attempt_idx: int | None = None,
    max_steps: int = 10,
    **reset_kwargs,
) -> ValidationBatch:    
    for rollout_idx in range(k):
        episode_env_spec = replace(env_spec, reset_args=tuple(reset_args), reset_kwargs=dict(reset_kwargs))
        episode_spec = EpisodeSpec(
            env=episode_env_spec,
            policy=policy_spec,
            candidate=candidate,

            task_id=task_id,
            attempt_idx=attempt_idx,
            rollout_idx=rollout_idx,

            trace_kind=trace_kind,
            max_steps=max_steps,
        )

        trace = runner.run(episode_spec)

def run_orchestrator_task(
    env_spec: EnvSpec,
    policy_spec: PolicySpec,
    runner: EpisodeRunner,
    designer: Designer,
    budget: BudgetPolicy,
    *reset_args,
    task_id: int | str = 0,
    task_description: str = "",
    history_traces: list[Trace] | None = None,
    validation_rollouts: int = 5,
    objective: MutationObjective | None = None,
    max_steps: int = 10,
    trace_store: TraceStore | None = None,
    **reset_kwargs,
) -> OrchestratorResult:
    history = list(history_traces or [])
    objective_signal = (
        objective.evaluate(history)
        if objective is not None
        else None
    )

    baseline_batch = _evaluate_candidate_k(
        env_spec,
        policy_spec,
        runner,
        Candidate(),
        validation_rollouts,
        *reset_args,
        trace_kind=TraceKind.BASELINE,
        task_id=task_id,
        max_steps=max_steps,
        **reset_kwargs,
    )    
    baseline = summarize_baseline(baseline_batch)
    if trace_store is not None:
        for trace in baseline_batch.traces:
            trace_store.add(trace)

    EnvCls = import_symbol(env_spec.import_path)
    if not issubclass(
        EnvCls,
        ActionableEnv,
    ):
        raise TypeError("EnvCls should be ActionableEnv")
    ctx = DesignerContext(
        history_traces=history,
        tool_schemas=EnvCls.tool_schemas(),
        env_state_schema=EnvCls.env_state_schema(),
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
        validation = _evaluate_candidate_k(
            env_spec,
            policy_spec,
            runner,
            proposal.candidate,
            validation_rollouts,
            *reset_args,
            trace_kind=TraceKind.EXPLORATION,
            task_id=task_id,
            attempt_idx=attempt_idx,
            max_steps=max_steps,
            **reset_kwargs,
        )

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

        if trace_store is not None:
            for trace in validation.traces:
                trace_store.add(trace)

        if decision.decision == Decision.ACCEPT:
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
    env_spec: EnvSpec,
    policy_spec: PolicySpec,
    runner: EpisodeRunner,
    designer: Designer,
    budget: BudgetPolicy,
    tasks: list[TaskSpec],
    *,
    validation_rollouts: int = 5,
    objective: MutationObjective | None = None,
    max_steps: int = 10,
    trace_store: TraceStore | None = None,
) -> OrchestratorRunResult:
    designer.reset()

    history: list[Trace] = []
    task_results = []

    for task in tasks:
        result = run_orchestrator_task(
            env_spec,
            policy_spec,
            runner,
            designer,
            budget,
            *task.reset_args,
            task_id=task.task_id,
            task_description=task.task_description,
            history_traces=history,
            validation_rollouts=validation_rollouts,
            objective=objective,
            max_steps=max_steps,
            trace_store=trace_store,
            **task.reset_kwargs,
        )

        task_results.append(result)
        history.extend(result.traces)

    return OrchestratorRunResult(
        task_results=task_results,
        history_traces=history,
    )
