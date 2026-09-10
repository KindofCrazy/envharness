import json

from envharness.core.types import Action, Candidate,  Diagnosis, DesignProposal
from envharness.agents.designer import Designer, DesignerContext, DesignProposal
from envharness.agents.prompts import build_propose_messages
from envharness.infra.llm import LLMClient

def parse_proposal_response(content: str) -> DesignProposal:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "designer response is not valid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "designer response must be a JSON object"
        )

    diagnosis = data.get("diagnosis", "")
    rules_code = data.get("rules_code", "")
    raw_actions = data.get("in_env_actions", [])

    if not isinstance(diagnosis, str):
        raise ValueError(
            "diagnosis must be a string"
        )

    if not isinstance(rules_code, str):
        raise ValueError(
            "rules_code must be a string"
        )

    if not isinstance(raw_actions, list):
        raise ValueError(
            "in_env_actions must be a list"
        )

    actions = []

    for item in raw_actions:
        if not isinstance(item, dict):
            raise ValueError(
                "each in_env_action must be an object"
            )

        name = item.get("name")
        kwargs = item.get("kwargs", {})

        if not isinstance(name, str):
            raise ValueError(
                "action name must be a string"
            )

        if not isinstance(kwargs, dict):
            raise ValueError(
                "action kwargs must be an object"
            )

        actions.append(
            Action(
                name=name,
                kwargs=dict(kwargs),
            )
        )

    return DesignProposal(
        diagnosis=Diagnosis(
            summary=diagnosis,
        ),
        candidate=Candidate(
            rules_code=rules_code,
            in_env_actions=actions,
        ),
    )


class LLMDesigner(Designer):

    def __init__(self, client: LLMClient, temperature: float = 0.2, max_tokens: int | None = None):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def propose(self, ctx: DesignerContext) -> DesignProposal:
        messages = build_propose_messages(ctx)
        response = self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return parse_proposal_response(response.content)

    def decide(self, candidate, validation, ctx):
        raise NotImplementedError(
            "LLMDesigner.decide is not implemented yet"
        )

    def refine(
        self,
        candidate,
        validation,
        ctx,
    ):
        raise NotImplementedError(
            "LLMDesigner.refine is not implemented yet"
        )  
