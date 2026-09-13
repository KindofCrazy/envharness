from typing import Any
from dataclasses import dataclass, field
from envharness.core.types import Action, Candidate, TraceKind

@dataclass
class EnvSpec:
    import_path: str
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    reset_args: tuple[Any, ...] = field(default_factory=tuple)
    reset_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicySpec:
    client_factory: str
    client_kwargs: dict[str, Any] = field(default_factory=dict)
    task_prompt: str = ""
    temperature: float = 0.4
    max_tokens: int | None = None

@dataclass
class EpisodeSpec:
    env: EnvSpec
    policy: PolicySpec
    candidate: Candidate = field(default_factory=Candidate)

    task_id = int | str | None = None
    attempt_idx: int | None = None
    rollout_idx: int | None = None

    trace_kind: TraceKind = TraceKind.EXPLORATION

    max_steps: int = 10

def episode_sepc_to_dict(spec: EpisodeSpec):
    return {
        "env": {
            "import_path": spec.env.import_path,
            "init_kwargs": spec.env.init_kwargs,
            "reset_args": spec.env.reset_args,
            "reset_kwargs": spec.env.reset_kwargs
        },
        "policy": {
            "client_factory": spec.policy.client_factory,
            "client_kwargs": spec.policy.client_kwargs,
            "task_prompt": spec.policy.task_prompt,
            "temperature": spec.policy.temperature,
            "max_tokens": spec.policy.max_tokens
        },
        "candidate": {
            "rules_code": spec.candidate.rules_code,
            "in_env_actions": [
            {
                "name": action.name,
                "kwargs": action.kwargs,
            } for action in spec.candidate.in_env_actions
            ]
        },

        "task_id": spec.task_id,
        "attempt_idx": spec.attempt_idx,
        "rollout_idx": spec.rollout_idx,
        "trace_kind": spec.trace_kind,
        "max_steps": spec.max_steps
        }

def episode_spec_from_dict(data):
    env = data["env"]
    policy = data["policy"]
    candidate = data["candidate"]

    return EpisodeSpec(
        env=EnvSpec(
            import_path=env["import_path"],
            init_kwargs=env["init_kwargs"],
            reset_args=env["reset_args"],
            reset_kwargs=env["reset_kwargs"],
        ),
        policy=PolicySpec(
            client_factory=policy["client_factory"],
            client_kwargs=policy["client_kwargs"],
            task_prompt=policy["task_prompt"],
            temperature=policy["temperature"],
            max_tokens=policy["max_tokens"],
        ),
        candidate=Candidate(
            rules_code=candidate["rules_code"],
            in_env_actions=[
                Action(
                    name=action["name"],
                    kwargs=action["kwargs"]
                ) for action in candidate["in_env_actions"]
            ]
        ),

        task_id=data["task_id"],
        attempt_idx=data["attempt_idx"],
        rollout_idx=data["rollout_idx"],
        trace_kind=data["trace_kind"],
        max_steps=data["max_steps"]
    )