import json

from envharness.core.types import Action, Trace, Candidate, ValidationBatch
from envharness.agents.designer import DesignerContext
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
description: {ctx.task_description}
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


def build_propose_messages(
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

def render_candidate(candidate: Candidate) -> str:
    actions = [
        {
            "name": action.name,
            "kwargs": action.kwargs,
        } 
        for action in candidate.in_env_actions
    ]

    return (
        "CANDIDATE\n"
        f"rules_code:\n{candidate.rules_code or '(empty)'}\n\n"
        "in_env_actions:\n"
        + json.dumps(
            actions,
            ensure_ascii=False,
            indent=2,
        )
    )

def render_validation(validation: ValidationBatch) -> str:
    parts = [
        "Validation",
        f"rollouts: {len(validation.traces)}",
        f"successes: {validation.success_count}",
        f"success_rate: {validation.success_rate}",
    ]

    for i, trace in enumerate(
        validation.traces
    ):
        parts.append(
            f"\nROLLOUT {i}\n"
            + render_trace(trace)
        )

    return "\n".join(parts)

def build_decide_messages(
    ctx: DesignerContext,
    candidate: Candidate,
    validation: ValidationBatch,
) -> list[Message]:
    context = render_designer_context(ctx)
    candidate_text = render_candidate(candidate)
    validation_text = render_validation(validation)

    user_prompt = f"""
{context}

{candidate_text}

{validation_text}

The candidate has now been evaluated.

Decide whether to:
- accept: keep this candidate
- refine: modify this candidate based on what was learned
- reject: abandon it and design a different candidate

Return JSON only:

{{
  "decision": "accept | refine | reject",
  "rationale": "short explanation",
  "failure_analysis": {{
    "primary_axis": "S0 | A | O | T | R | task_understanding | none",
    "label": "short failure label",
    "description": "what went wrong"
  }}
}}

failure_analysis may be null when there is no meaningful failure.
"""

    return [
        Message(
            role="system",
            content=DESIGNER_SYSTEM_PROMPT.strip(),
        ),
        Message(
            role="user",
            content=user_prompt.strip(),
        ),
    ]

def build_refine_messages(
    ctx: DesignerContext,
    candidate: Candidate,
    validation: ValidationBatch
) -> list[Message]:
    context = render_designer_context(ctx)

    user_prompt = f"""
{context}

{render_candidate(candidate)}

{render_validation(validation)}

Refine the previous candidate based on these rollout results.

Preserve useful parts of the previous mutation when appropriate,
but change it if the evidence shows it is ineffective or harmful.

Return JSON only, with this shape:

{{
  "diagnosis": "what should change and why",
  "rules_code": "complete Python source or empty string",
  "in_env_actions": [
    {{
      "name": "tool name",
      "kwargs": {{}}
    }}
  ]
}}
"""
