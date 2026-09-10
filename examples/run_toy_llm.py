import os

from pathlib import Path
from envharness.bridges.toy24.bridge import Toy24Env
from envharness.infra.llm import OpenAICompatibleClient
from envharness.agents.llm_policy import LLMPolicy
from envharness.agents.llm_designer import LLMDesigner
from envharness.orchestration.budget import FixedBudget
from envharness.orchestration.orchestrator import (
    TaskSpec,
    run_orchestrator,
)

api_key_path = Path(__file__).parent / "api_key.txt"

api_key = api_key_path.read_text(
    encoding="utf-8"
).strip()

policy_client = OpenAICompatibleClient(
    model_id="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    thinking=False
)

designer_client = OpenAICompatibleClient(
    model_id="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    thinking=False
)

env = Toy24Env()

policy = LLMPolicy(
    client=policy_client,
    tool_schemas=env.tool_schemas(),
    task_prompt=(
        "Solve the Toy24 task. "
        "Use combine to combine two current numbers. "
        "The op argument must be one of: "
        "add, sub, mul, div. "
        "When a current number equals the target, "
        "call stop."
    ),
    temperature=0.0,
)

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
    base_env=env,
    policy=policy,
    designer=designer,
    budget=budget,
    tasks=tasks,
    validation_rollouts=2,
    max_steps=8,
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