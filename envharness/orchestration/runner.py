from envharness.core.actionable_env import ActionableEnv
from envharness.agents.policy import Policy
from envharness.core.types import Step, Trace

def run_episode(
    env: ActionableEnv,
    policy: Policy,
    *reset_args,
    **reset_kwargs,
) -> Trace:
    policy.reset()
    observation = initial_observation = env.reset(*reset_args, **reset_kwargs)

    trace = Trace(
        initial_observation=initial_observation,
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
    
