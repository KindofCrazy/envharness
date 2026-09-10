from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.core.types import Step, Trace, TraceKind

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
    observation = initial_observation = env.reset(*reset_args, **reset_kwargs)

    trace = Trace(
        initial_observation=initial_observation,
        kind = trace_kind,
        task_id=task_id,
        attempt_idx=attempt_idx,
        rollout_idx=rollout_idx
    )

    while True:
        action = policy.act(observation)
        response = env.step(action)
        step = Step(
            observation=observation,
            action=action,
            response=response
        )
        trace.steps.append(step)

        observation = response.observation
        if response.terminated or response.truncated:
            break

    evaluation = env.evaluate()
    trace.success = evaluation.success
    return trace

