from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.core.types import Step, Trace, TraceKind, Observation

def run_episode(
    env: ActionableEnv,
    policy: Policy,
    *reset_args,
    trace_kind: TraceKind = TraceKind.EXPLORATION,
    task_id: int | str | None = None,
    attempt_idx: int | None = None,
    rollout_idx: int | None = None,
    **reset_kwargs,
) -> Trace:
    policy.reset()
    try:
        observation = initial_observation = env.reset(*reset_args, **reset_kwargs)
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

    while True:
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

