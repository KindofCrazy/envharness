from envharness.core.actionable_env import ActionableEnv
from envharness.core.types import Candidate
from envharness.harness.setup import Setup
from envharness.core.code_loader import load_rules_subclass
from envharness.orchestration.specs import EnvSpec, EpisodeSpec, PolicySpec
from envharness.infra.utils import import_symbol
from envharness.agents.llm_policy import LLMClient, LLMPolicy

def build_env_stack(
    base: ActionableEnv,
    candidate: Candidate
) -> ActionableEnv:
    env = base

    if candidate.in_env_actions:
        env = Setup(inner=env, actions=list(candidate.in_env_actions))

    if candidate.rules_code.strip():
        RulesCls = load_rules_subclass(candidate.rules_code)
        rules = RulesCls(inner=env)
        rules.rules_code = candidate.rules_code
        env = rules

    return env

def build_base_env(
    spec: EnvSpec,
) -> ActionableEnv:
    EnvCls = import_symbol(spec.import_path)
    env = EnvCls(**spec.init_kwargs)

    if not isinstance(
        env,
        ActionableEnv,
    ):
        raise TypeError(
            "EnvSpec did not build "
            "an ActionableEnv"
        )

    return env

def build_episode_env(
    spec: EpisodeSpec
) -> ActionableEnv:
    base = build_base_env(spec)

    return build_env_stack(
        base, spec.candidate
    )

def build_policy(
    spec: PolicySpec,
    tool_schemas: list[dict]
) -> LLMPolicy:

    ClientCls = import_symbol(spec.client_factory)
    client = ClientCls(**spec.client_factory)

    if not isinstance(
        client,
        LLMClient,
    ):
        raise TypeError(
            "PolicySpec client_factory "
            "did not build an LLMClient"
        )

    return LLMPolicy(
        client=client,
        tool_schemas=tool_schemas,
        task_prompt=spec.task_prompt,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
    ) 