import json
from dataclasses import dataclass, field
from envharness.core.types import Trace
from envharness.infra.llm import Message, LLMClient

@dataclass
class SkillDraft:
    title: str
    description: str
    content: str
    source: dict = field(default_factory=dict)

def format_trace(trace: Trace) -> str:
    lines = [
        f"kind: {trace.kind.value}",
        f"success: {trace.success}",
        f"error: {trace.error or 'none'}",
        (
            "initial observation: "
            f"{trace.initial_observation.text}"
        ),
    ]

    for i, step in enumerate(trace.steps):
        action = json.dumps(
            {
                "name": step.action.name,
                "kwargs": step.action.kwargs,
            },
            ensure_ascii=False,
        )

        lines.extend([
            f"step {i}:",
            f"  observation: {step.observation.text}",
            f"  action: {action}",
            (
                "  next observation: "
                f"{step.response.observation.text}"
            ),
            f"  reward: {step.response.reward}",
            (
                "  terminated: "
                f"{step.response.terminated}"
            ),
            (
                "  truncated: "
                f"{step.response.truncated}"
            ),
        ])

    return "\n".join(lines)

SKILL_INDUCTION_SYSTEM_PROMPT = """
You extract reusable problem-solving skills from agent trajectories.

Analyze the trajectories together.

Infer general lessons that would help an agent solve similar tasks
in the future.

Successful trajectories show useful strategies.
Failed trajectories may reveal mistakes to avoid.

A trace with a runtime/framework error is not evidence that the
agent's task-solving strategy was wrong. Do not derive task skills
from infrastructure failures.

Each skill must be generalizable. Do not merely memorize one exact
action sequence or one task answer.

Return JSON only:

{
  "skills": [
    {
      "title": "...",
      "description": "...",
      "content": "..."
    }
  ]
}
""".strip()

def build_induction_messages(
    task_description: str,
    traces: list[Trace],
) -> list[Message]:

    usable = [
        trace
        for trace in traces
        if trace.error is None
    ]

    if usable == []:
        return []

    trajectories = []

    for i, trace in enumerate(usable):
        trajectories.append(
            f"TRAJECTORY {i}\n"
            + format_trace(trace)
        )

    user = (
        f"TASK\n{task_description}\n\n"
        + "\n\n".join(trajectories)
    )

    return [
        Message(
            role="system",
            content=SKILL_INDUCTION_SYSTEM_PROMPT,
        ),
        Message(
            role="user",
            content=user,
        ),
    ]

def parse_skill_response(
    content: str,
    *,
    max_items: int = 3,
) -> list[SkillDraft]:
    data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError("data must be dict")

    skills = data.get("skills", [])

    if not isinstance(skills, list):
        raise ValueError("skills must be list")

    skill_drafts = []
    for skill in skills:
        if not isinstance(skill, dict):
            raise ValueError(
                "each skill must be an object"
            )

        title = skill.get("title")
        description = skill.get("description")
        content = skill.get("content")

        if not isinstance(title, str):
            raise ValueError(
                "skill title must be a string"
            )

        if not isinstance(description, str):
            raise ValueError(
                "skill description must be a string"
            )

        if not isinstance(content, str):
            raise ValueError(
                "skill content must be a string"
            )

        skill_drafts.append(
            SkillDraft(
                title=title,
                description=description,
                content=content,
            )
        )

    return skill_drafts[:max_items]


class LLMSkillInducer:

    def __init__(self, client: LLMClient, *, max_items: int = 3, temperature: float = 0.0):
        self.client = client
        self.max_items = max_items
        self.temperature = temperature

    def induce(self, task_description: str, traces: list[Trace],) -> list[SkillDraft]:
        usable = [trace for trace in traces if trace.error is None]

        if not usable:
            return []

        messages = build_induction_messages(task_description, usable)

        response = self.client.chat(messages, temperature=self.temperature)

        skills = parse_skill_response(response.content, max_items=self.max_items)

        source = {
            "task_ids": sorted({str(trace.task_id) for trace in usable if trace.task_id is not None}),
            "trace_count": len(usable),
            "success_count": sum(trace.success for trace in usable),
        }

        for skill in skills:
            skill.source = dict(source)

        return skills
