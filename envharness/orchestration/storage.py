import json
from pathlib import Path

from envharness.core.types import Observation, Action, EnvResponse, Step, TraceKind, Trace

def _observation_to_dict(obs: Observation) -> dict:
    return {
        "text": obs.text,
        "data": obs.data,
    }


def _observation_from_dict(data: dict) -> Observation:
    return Observation(
        text=data["text"],
        data=dict(data.get("data", {})),
    )

def _action_to_dict(action: Action) -> dict:
    return {
        "name": action.name,
        "kwargs": action.kwargs,
    }


def _action_from_dict(data: dict) -> Action:
    return Action(
        name=data["name"],
        kwargs=dict(data.get("kwargs", {})),
    )

def _response_to_dict(
    response: EnvResponse,
) -> dict:
    return {
        "observation": _observation_to_dict(
            response.observation
        ),
        "reward": response.reward,
        "terminated": response.terminated,
        "truncated": response.truncated,
        "info": response.info,
    }

def _response_from_dict(data: dict) -> EnvResponse:
    return EnvResponse(
        observation=_observation_from_dict(data["observation"]),
        reward=data["reward"],
        terminated=data["terminated"],
        truncated=data["truncated"],
        info=dict(data.get("info", {})),
    )

def trace_to_dict(trace: Trace) -> dict:
    return {
        "initial_observation": _observation_to_dict(
            trace.initial_observation
        ),
        "steps": [
            {
                "observation": _observation_to_dict(
                    step.observation
                ),
                "action": _action_to_dict(
                    step.action
                ),
                "response": _response_to_dict(
                    step.response
                ),
            }
            for step in trace.steps
        ],
        "success": trace.success,
        "kind": trace.kind.value,
        "task_id": trace.task_id,
        "attempt_idx": trace.attempt_idx,
        "rollout_idx": trace.rollout_idx,
        "error": trace.error,
    }

def trace_from_dict(data: dict) -> Trace:
    return Trace(
        initial_observation=_observation_from_dict(
            data["initial_observation"]
        ),
        steps=[
            Step(
                observation=_observation_from_dict(
                    step["observation"]
                ),
                action=_action_from_dict(
                    step["action"]
                ),
                response=_response_from_dict(
                    step["response"]
                ),
            )
            for step in data.get("steps", [])
        ],
        success=data.get("success", False),
        kind=TraceKind(
            data.get(
                "kind",
                TraceKind.EXPLORATION.value,
            )
        ),
        task_id=data.get("task_id"),
        attempt_idx=data.get("attempt_idx"),
        rollout_idx=data.get("rollout_idx"),
        error=data.get("error"),
    )


class TraceStore:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._cache: list[Trace] = []

        if self.path.exists():
            with self.path.open(
                encoding="utf-8"
            ) as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    self._cache.append(
                        trace_from_dict(
                            json.loads(line)
                        )
                    )

    def add(self, trace: Trace) -> None:
        self._cache.append(trace)

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    trace_to_dict(trace),
                    ensure_ascii=False,
                )
                + "\n"
            )

    def all(self) -> list[Trace]:
        return list(self._cache)

    def recent(self, n: int) -> list[Trace]:
        return self._cache[-n:]

    def filter_kind(
        self,
        kind: TraceKind,
    ) -> list[Trace]:
        return [
            trace
            for trace in self._cache
            if trace.kind == kind
        ]

    def __len__(self) -> int:
        return len(self._cache)
