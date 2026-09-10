import json

from envharness.core.types import Action, Candidate, Diagnosis, DecideResult, Decision, FailureAnalysis, DesignProposal
from envharness.agents.designer import Designer, DesignerContext
from envharness.agents.prompts import build_propose_messages, build_decide_messages, build_refine_messages
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

def parse_decide_response(
    content: str,
) -> DecideResult:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "designer decide response "
            "is not valid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "designer decide response "
            "must be a JSON object"
        )

    raw_decision = data.get("decision")

    try:
        decision = Decision(raw_decision)
    except ValueError as exc:
        raise ValueError(
            f"invalid decision: {raw_decision}"
        ) from exc

    rationale = data.get(
        "rationale",
        "",
    )

    if not isinstance(rationale, str):
        raise ValueError(
            "rationale must be a string"
        )

    raw_failure = data.get(
        "failure_analysis"
    )

    failure_analysis = None

    if raw_failure is not None:
        if not isinstance(
            raw_failure,
            dict,
        ):
            raise ValueError(
                "failure_analysis "
                "must be an object or null"
            )

        axis = raw_failure.get(
            "primary_axis"
        )

        valid_axes = {
            "S0",
            "A",
            "O",
            "T",
            "R",
            "task_understanding",
            "none",
        }

        if axis not in valid_axes:
            raise ValueError(
                f"invalid primary_axis: {axis}"
            )

        failure_analysis = FailureAnalysis(
            primary_axis=axis,
            label=str(
                raw_failure.get(
                    "label",
                    "",
                )
            ),
            description=str(
                raw_failure.get(
                    "description",
                    "",
                )
            ),
        )

    return DecideResult(
        decision=decision,
        failure_analysis=failure_analysis,
        rationale=rationale,
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
        messages = build_decide_messages(
            ctx=ctx,
            candidate=candidate,
            validation=validation
        )

        response = self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return parse_decide_response(response.content)

    def refine(
        self,
        candidate,
        validation,
        ctx,
    ):
        messages = build_refine_messages(
            ctx,
            candidate,
            validation,
        )

        response = self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return parse_proposal_response(
            response.content
        )
