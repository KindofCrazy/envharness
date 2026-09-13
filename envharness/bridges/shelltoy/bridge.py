import sys
import json
import subprocess

from dataclasses import dataclass
from envharness.core.actionable_env import ActionableEnv
from envharness.bridges.shelltoy.tools import Do
from envharness.core.types import Action, EnvResponse, EnvResetResponse, Observation, EvaluationResult

@dataclass
class ShellToyState:
    value: int = 0
    target: int = 24
    stopped: bool = False
    success: bool = False
    step_count: int = 10
    last_message: str = ""


class ShellToyEnv(ActionableEnv):

    tool_registry = [Do]

    def __init__(self):
        self._proc = None
        self.state = ShellToyState()

    def _start_runtime(self):
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "envharness.bridges.shelltoy.runtime"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def _request(self, payload: dict) -> dict:
        if self._proc is None:
            raise RuntimeError("runtime is not started")

        line = json.dumps(payload)
        self._proc.stdin.write(line + "\n")

        response_line = self._proc.stdout.readline()
        if not response_line:
            raise RuntimeError("runtime exited unexpectedly")

        return json.loads(response_line)

    def _update_state(self, state: dict):
        self.state = ShellToyState(
            value=state["value"],
            target=state["target"],
            stopped=state["stopped"],
            success=state["success"],
            step_count=state["step_count"],
            last_message=state["last_message"]
        )

    def reset(self, value: int = 1, target: int = 24) -> EnvResetResponse:
        if self._proc is not None:
            self.close()

        self._start_runtime()

        result = self._request({
            "op": "reset",
            "value": value,
            "target": target,
        })

        self._update_state(result)

        return EnvResetResponse(
            observation=self.observe(),
            info={}
        )


    def step(self, action: Action) -> EnvResponse:
        self.state.step_count += 1

        if action.name != "do":
            raise ValueError("action.name should only be 'do'")

        text = action.kwargs.get("text")
        result = self._request({
            "op": "command",
            "text": text,
        })

        self._update_state(result)

    def observe(self) -> Observation:
        return Observation(
            text=(
                f"value={self.state.value}, "
                f"target={self.state.target}, "
                f"last={self.state.last_message}"
            ),
            data={
                "value": self.state.value,
                "target": self.state.target,
                "stopped": self.state.stopped,
                "step_count": self.state.step_count,
            },
        )

    def evaluate(self):
        return EvaluationResult(
            success=(
                self.state.stopped
                and self.state.success
            ),
            score=(
                1.0
                if self.state.stopped
                and self.state.success
                else 0.0
            ),
        )

    def get_env_state(self):
        return self.state

    def close(self):
        if self._proc is None:
            return

        proc = self._proc
        self._proc = None

        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(
                        '{"op":"close"}\n'
                    )
                    proc.stdin.flush()
                except Exception:
                    pass

                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        finally:
            proc.kill()
            proc.wait()

    def save_state(self):
        return {
            "value": self.state.value,
            "target": self.state.target,
            "stopped": self.state.stopped,
            "success": self.state.success,
            "step_count": self.state.step_count,
            "last_message": self.state.last_message
        }

    @classmethod
    def from_state(cls, state):
        env = cls()
        env.state = ShellToyState(
            value=state["value"],
            target=state["target"],
            stopped=state["stopped"],
            success=state["success"],
            step_count=state["step_count"],
            last_message=state["last_message"]
        )

        return env

    def reset_after_load(self):
        return True

    