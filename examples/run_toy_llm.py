import os
from pathlib import Path

from envharness.infra.llm import OpenAICompatibleClient
from envharness.agents.llm_designer import LLMDesigner
from envharness.orchestration.budget import FixedBudget
from envharness.orchestration.orchestrator import (
    TaskSpec,
    run_orchestrator,
)
from envharness.orchestration.runner import InProcessRunner
from envharness.orchestration.specs import EnvSpec, PolicySpec
from envharness.orchestration.storage import TraceStore

store = TraceStore(
    Path("runs/toy24/traces.jsonl")
)

api_key = os.environ["DEEPSEEK_API_KEY"]

designer_client = OpenAICompatibleClient(
    model_id="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    thinking=False,
)

env_spec = EnvSpec(
    import_path=(
        "envharness.bridges.toy24.bridge:"
        "Toy24Env"
    ),
)

policy_spec = PolicySpec(
    client_factory=(
        "envharness.infra.llm:"
        "OpenAICompatibleClient"
    ),
    client_kwargs={
        "model_id": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": api_key,
        "thinking": False,
    },
    task_prompt=(
        "Solve the Toy24 task..."
    ),
    temperature=0.0,
)

runner = InProcessRunner()

designer = LLMDesigner(
    client=designer_client,
    temperature=0.0,
)

budget = FixedBudget(
    max_attempts=2,
)

tasks = [
    TaskSpec(
        task_id="toy24-smoke",
        task_description=(
            "Adapt this Toy24 environment based on "
            "the policy baseline and rollout behavior. "
            "Keep the task solvable while creating "
            "a meaningful challenge."
        ),
        reset_args=([1, 2, 3, 4],),
        reset_kwargs={
            "target": 24,
        },
    )
]

result = run_orchestrator(
    env_spec=env_spec,
    policy_spec=policy_spec,
    runner=runner,
    designer=designer,
    budget=budget,
    tasks=tasks,
    validation_rollouts=2,
    max_steps=8,
    trace_store=store,
)

task_result = result.task_results[0]

print(
    "baseline success rate:",
    task_result.baseline.success_rate,
)

for i, attempt in enumerate(
    task_result.attempts
):
    print()
    print("attempt:", i)
    print(
        "diagnosis:",
        attempt.proposal.diagnosis.summary,
    )
    print(
        "candidate:",
        attempt.proposal.candidate,
    )
    print(
        "validation SR:",
        attempt.validation.success_rate,
    )
    print(
        "decision:",
        attempt.decision.decision,
    )
    print(
        "rationale:",
        attempt.decision.rationale,
    )

    for trace in attempt.validation.traces:
        print(
            "  trace:",
            trace.success,
            trace.error,
        )

print()
print(
    "accepted:",
    task_result.accepted_candidate,
)

print()
print("stored traces:", len(store))
