import json
from pathlib import Path
import subprocess
import sys
import tempfile

from envharness.core.types import Trace, Observation
from envharness.orchestration.runner import EpisodeRunner, run_episode_spec
from envharness.orchestration.specs import EpisodeSpec, episode_spec_to_dict, episode_spec_from_dict
from envharness.orchestration.storage import trace_from_dict, trace_to_dict

def make_failure_trace(spec: EpisodeSpec, error: str) -> Trace:
    return Trace(
        initial_observation=Observation(
            text="episode failed"
        ),
        kind=spec.trace_kind,
        task_id=spec.task_id,
        attempt_idx=spec.attempt_idx,
        rollout_idx=spec.rollout_idx,
        error=error
    )

class SubprocessRunner(EpisodeRunner):

    def __init__(self, timeout_seconds: float = 120.0):
        self.timeout_seconds = timeout_seconds

    def run(self, spec: EpisodeSpec):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            input_path = tmp_dir / "input.json"
            output_path = tmp_dir / "output.json"

            input_path.write_text(
                json.dumps(episode_spec_to_dict(spec)), encoding="utf-8"
            )

            command = [
                sys.executable,
                "-m",
                "envharness.orchestration.worker",
                str(input_path),
                str(output_path),
            ]

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )

            except subprocess.TimeoutExpired:
                return make_failure_trace(spec, "episode subprocess timed out")

            if output_path.exists():
                try:
                    data = json.loads(output_path.read_text(encoding="utf-8"))
                    return trace_from_dict(data)

                except Exception as exc:
                    return make_failure_trace(spec, f"invalid subprocess\n output: {exc}")
                
            if result.returncode != 0:
                return make_failure_trace(
                    spec,
                    (
                        "episode subprocess "
                        "failed: "
                        f"{result.stderr[-2000:]}"
                    ),
                )

            return make_failure_trace(
                spec,
                "episode subprocess produced no output",
            )


def main():
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    data = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    spec = episode_spec_from_dict(
        data
    )

    trace = run_episode_spec(
        spec
    )

    output_path.write_text(
        json.dumps(
            trace_to_dict(trace),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()