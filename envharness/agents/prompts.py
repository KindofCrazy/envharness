import json

from envharness.core.types import Action, Trace
from envharness.agents.designer import Designer, DesignerContext
from envharness.infra.llm import Message

DESIGNER_SYSTEM_PROMPT = """
You are an environment designer.

Your job is to modify the environment experienced by a policy
based on its task, trajectories, baseline performance, and objective.

A Candidate has two mutation mechanisms:

1. in_env_actions
   Actions replayed before the policy starts.
   Use only tools listed in the tool schemas.

2. rules_code
   Python source defining:

       class _Rules(Rules):

   You may override:

       filter_action(action, env_state)
       modify_transition(action, response, env_state)
       filter_observation(observation, env_state)

The env_state argument has exactly the structure described
in the environment-state schema.

Do not invent tool names or env_state fields.
"""

def render_action(action: Action) -> str:
    return json.dumps(
        {
            "name": action.name,
            "kwargs": action.kwargs
        },
        ensure_ascii=False,
    )

def render_trace(trace: Trace) -> str:
    lines = [
        (
            f"kind={trace.kind.value}, "
            f"task_id={trace.task_id}, "
            f"attempt_idx={trace.attempt_idx}, "
            f"rollout_idx={trace.rollout_idx}, "
            f"success={trace.success}"
        ),
        f"initial: {trace.initial_observation.text}",
    ]

    for i, step in enumerate(trace.steps):
        lines.append(
            f"step {i}:"
        )
        lines.append(
            f"  observation: {step.observation.text}"
        )
        lines.append(
            f"  action: {render_action(step.action)}"
        )
        lines.append(
            f"  response: {step.response.observation.text}"
        )
        lines.append(
            f"  terminated={step.response.terminated}, "
            f"truncated={step.response.truncated}"
        )

    return "\n".join(lines)

def render_designer_context(ctx: DesignerContext) ->str:
    parts = []
    parts.append(
        f"""TASK
task_id: {ctx.task_id}
description
"""
    )

    parts.append(
        "TOOLS\n"
        + json.dumps(
            ctx.tool_schemas,
            ensure_ascii=False,
            indent = 2
        )
    )

    parts.append(
        "ENVIRONMENT STATE\n"
        + ctx.env_state_schema
    )

    if ctx.baseline is not None:
        parts.append(
            f"""BASELINE
rollouts: {ctx.baseline.n}
successes: {ctx.baseline.n_success}
success_rate: {ctx.baseline.success_rate}
avg_success_steps: {ctx.baseline.avg_success_steps}
"""
    )

    if ctx.objective_signal is not None:
        parts.append(
            f"""OBJECTIVE
score: {ctx.objective_signal.score}
diagnostic: {ctx.objective_signal.diagnostic}
suggestion: {ctx.objective_signal.suggestion}
"""
        )

    if ctx.history_traces:
        history_text = "\n\n".join(
            render_trace(trace)
            for trace in ctx.history_traces
        )

        parts.append(
            "HISTORY\n" + history_text
        )
    else:
        parts.append(
            "HISTORY\n(no previous traces)"
        )

    return "\n\n".join(parts)


def build_propose_message(
    ctx: DesignerContext
) -> list[Message]:
    context = render_designer_context(ctx)

    user_prompt = f"""
{context}

Propose one environment mutation.

Return JSON only, with this shape:

{{
  "diagnosis": "short explanation of the policy weakness",
  "rules_code": "Python source or empty string",
  "in_env_actions": [
    {{
      "name": "tool name",
      "kwargs": {{}}
    }}
  ]
}}
"""

    return [
        Message(
            role="system",
            content=DESIGNER_SYSTEM_PROMPT.strip()
        ),
        Message(
            role="user",
            content=user_prompt.strip()
        )
    ]