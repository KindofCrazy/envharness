import json
from pathlib import Path
import subprocess
import sys
import tempfile

from envharness.orchestration.runner import run_episode_spec
from envharness.orchestration.specs import episode_spec_from_dict
from envharness.orchestration.storage import trace_to_dict

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