from abc import ABC, abstractmethod
from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.core.types import Step, Trace, TraceKind, Observation
from envharness.orchestration.specs import EpisodeSpec
from envharness.orchestration.builder import build_episode_env, build_policy
from envharness.infra.utils import import_symbol

def run_episode(
    env: ActionableEnv,
    policy: Policy,
    *reset_args,
    trace_kind: TraceKind = TraceKind.EXPLORATION,
    task_id: int | str | None = None,
    attempt_idx: int | None = None,
    rollout_idx: int | None = None,
    max_steps: int = 10,
    **reset_kwargs,
) -> Trace:
    policy.reset()
    try:
        reset_response = env.reset(
            *reset_args,
            **reset_kwargs,
        )

        observation = (
            reset_response.observation
        )

        initial_observation = observation    
    except Exception as exc:
        return Trace(
        initial_observation=Observation(
            text="episode failed before reset completed"
        ),
        kind=trace_kind,
        task_id=task_id,
        attempt_idx=attempt_idx,
        rollout_idx=rollout_idx,
        error=(
            "env.reset raised: "
            f"{type(exc).__name__}: {exc}"
        ),
    )

    trace = Trace(
        initial_observation=initial_observation,
        kind=trace_kind,
        task_id=task_id,
        attempt_idx=attempt_idx,
        rollout_idx=rollout_idx
    )

    for _ in range(max_steps):
        try:
            action = policy.act(observation)
        except Exception as exc:
            trace.error = (
                "policy.act raised: "
                f"{type(exc).__name__}: {exc}"
            )
            return trace

        try:
            response = env.step(action)
        except Exception as exc:
            trace.error = (
                "env.step raised: "
                f"{type(exc).__name__}: {exc}"
            )
            return trace

        step = Step(
            observation=observation,
            action=action,
            response=response
        )
        trace.steps.append(step)

        observation = response.observation
        if response.terminated or response.truncated:
            break

        if len(trace.steps) >= max_steps:
            response.truncated = True
            break

    try:
        evaluation = env.evaluate()
    except Exception as exc:
        trace.error = (
            "env.evaluate raised: "
            f"{type(exc).__name__}: {exc}"
        )
        return trace

    trace.success = evaluation.success
    return trace

def run_episode_spec(spec: EpisodeSpec) -> Trace:
    env = None

    try:
        try:
            env = build_episode_env(spec)
        except Exception as exc:
            return Trace(
                initial_observation=Observation(
                    text="episode failed before environment construction completed"
                ),
                kind=spec.trace_kind,
                task_id=spec.task_id,
                attempt_idx=spec.attempt_idx,
                rollout_idx=spec.rollout_idx,
                error=(
                    "episode build failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        EnvCls = import_symbol(
            spec.env.import_path
        )

        policy = build_policy(
            spec.policy,
            EnvCls.tool_schemas(),
        )

        return run_episode(
            env,
            policy,
            *spec.env.reset_args,
            trace_kind=spec.trace_kind,
            task_id=spec.task_id,
            attempt_idx=spec.attempt_idx,
            rollout_idx=spec.rollout_idx,
            max_steps=spec.max_steps,
            **spec.env.reset_kwargs,
        )

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


class EpisodeRunner(ABC):

    @abstractmethod
    def run(self, spec: EpisodeSpec) -> Trace:
        ...

class InProcessRunner(EpisodeRunner):

    def run(self, spec: EpisodeSpec) -> Trace:
        return run_episode_spec(spec)

    