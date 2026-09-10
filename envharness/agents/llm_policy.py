import json

from envharness.agents.policy import Policy
from envharness.core.types import Action, Observation
from envharness.infra.llm import LLMClient, Message

def parse_action_response(
    content: str,
) -> Action:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "policy response is not valid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "policy response must be a JSON object"
        )

    name = data.get("name")
    kwargs = data.get("kwargs", {})

    if not isinstance(name, str):
        raise ValueError(
            "policy action name must be a string"
        )

    if not isinstance(kwargs, dict):
        raise ValueError(
            "policy action kwargs must be an object"
        )

    return Action(
        name=name,
        kwargs=dict(kwargs),
    )

class LLMPolicy(Policy):

    def __init__(
        self,
        client: LLMClient,
        tool_schemas: list[dict],
        task_prompt: str = "",
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ):
        self.client = client
        self.tool_schemas = list(tool_schemas)
        self.task_prompt = task_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: list[Message] = []

    def reset(self):
        self.history = []

    def _system_prompt(self) -> str:
        return f"""
You are an agent interacting with an environment.

TASK:
{self.task_prompt}

AVAILABLE TOOLS:
{json.dumps(self.tool_schemas, indent=2)}

At every step, choose exactly one action.

Return JSON only:

{{
  "name": "tool name",
  "kwargs": {{}}
}}

Use only a tool listed above.
""".strip()

    def act(self, observation: Observation) -> Action:
        if not self.history:
            self.history.append(
                Message(
                    role="system",
                    content=self._system_prompt
                )
            )

        self.history.append(
            Message(
                role="user",
                content=observation.text
            )
        )

        response = self.client.chat(
            messages=list(self.history),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return parse_action_response(response.content)