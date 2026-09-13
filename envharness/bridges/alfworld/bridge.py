import os

from dataclasses import dataclass, field
from typing import Any

from envharness.core.actionable_env import ActionableEnv
from envharness.core.registry import register_env

from envharness.core.types import (
    Action,
    EnvResetResponse,
    EnvResponse,
    EvaluationResult,
    Observation,
)

from envharness.bridges.alfworld.tools import Do


@dataclass
class AlfworldEnvState:
    """
    Pure-data view of the current ALFWorld world state.

    Important:
    this state is visible to Rules.
    It must NOT contain the real TextWorld runtime object.
    """

    goal_text: str = ""
    obs_text: str = ""

    admissible_commands: list[str] = field(
        default_factory=list
    )

    score: float = 0.0

    won: bool = False
    done: bool = False

    step_count: int = 0


@register_env("alfworld")
class AlfworldEnv(ActionableEnv):

    tool_registry = [Do]

    def __init__(self):
        # Real ALFWorld/TextWorld runtime.
        #
        # This is PRIVATE runtime state.
        # Rules must never see this object.
        self._env = None

        self._current_split: str | None = None

        # Pure data mirror exposed through get_env_state().
        self.state = AlfworldEnvState()

    # ============================================================
    # Runtime construction
    # ============================================================

    def _lazy_init_env(
        self,
        config_path: str,
        split: str,
    ) -> None:
        """
        Construct the real ALFWorld TextWorld environment.

        Imports stay inside this method so importing EnvHarness
        does not require ALFWorld to be installed.
        """

        try:
            import yaml
            import alfworld.agents.environment as env_module
        except ImportError as exc:
            raise RuntimeError(
                "ALFWorld is not installed. "
                'Install it with pip install "alfworld[full]".'
            ) from exc

        if not os.path.exists(config_path):
            raise RuntimeError(
                f"ALFWorld config does not exist: {config_path}"
            )

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:
            config = yaml.safe_load(f)

        # Get ALFWorld's TextWorld backend.
        env_cls = env_module.get_environment(
            "AlfredTWEnv"
        )

        raw_env = env_cls(
            config,
            train_eval=split,
        )

        # ALFWorld itself exposes a batched interface.
        # We use batch_size=1 and remove that batch dimension
        # inside the Bridge.
        self._env = raw_env.init_env(
            batch_size=1
        )

        self._current_split = split

    # ============================================================
    # Small conversion helpers
    # ============================================================

    @staticmethod
    def _first(value):
        """
        Remove the batch_size=1 dimension.

        Example:

            ["hello"] -> "hello"

        but scalar values remain unchanged.
        """

        if (
            isinstance(value, (list, tuple))
            and len(value) == 1
        ):
            return value[0]

        return value

    @classmethod
    def _unwrap_info(
        cls,
        info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        ALFWorld returns batched info values.

        Convert:

            {
                "won": [False],
                "admissible_commands": [
                    ["look", "go to table 1"]
                ]
            }

        into:

            {
                "won": False,
                "admissible_commands": [
                    "look",
                    "go to table 1"
                ]
            }
        """

        if not isinstance(info, dict):
            return {}

        result = {}

        for key, value in info.items():
            result[key] = cls._first(value)

        return result

    @staticmethod
    def _as_string_list(
        value,
    ) -> list[str]:
        """
        Normalize a possible ALFWorld value into list[str].
        """

        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        if isinstance(value, (list, tuple)):
            return [
                str(item)
                for item in value
            ]

        return [str(value)]

    @staticmethod
    def _extract_goal(
        obs_text: str,
    ) -> str:
        """
        Minimal goal extraction.

        ALFWorld initial observations commonly contain a line like:

            Your task is to: put the apple in the fridge.

        We deliberately keep this parser simple for now.
        """

        marker = "Your task is to:"

        if marker not in obs_text:
            return ""

        after = obs_text.split(
            marker,
            1,
        )[1]

        # Usually the goal itself is one line.
        line = after.strip().splitlines()[0]

        return line.strip()

    # ============================================================
    # ActionableEnv lifecycle
    # ============================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        split: str = "train",
        config_path: str | None = None,
    ) -> EnvResetResponse:
        """
        Reset one ALFWorld episode.

        Minimal supported configuration:

            seed
            split
            config_path

        More advanced official options such as task pinning,
        subsets and replay are intentionally omitted.
        """

        if config_path is None:
            config_path = os.environ.get(
                "ALFWORLD_CONFIG"
            )

        if not config_path:
            raise RuntimeError(
                "ALFWorld config path is required. "
                "Pass config_path=... to reset() "
                "or set the ALFWORLD_CONFIG environment variable."
            )

        # Rebuild if:
        #   - this is the first reset
        #   - user switched train/eval split
        if (
            self._env is None
            or split != self._current_split
        ):
            self.close()

            self._lazy_init_env(
                config_path=config_path,
                split=split,
            )

        # ALFWorld/TextWorld supports seeding.
        if seed is not None:
            seed_fn = getattr(
                self._env,
                "seed",
                None,
            )

            if callable(seed_fn):
                seed_fn(seed)

        # Real ALFWorld call.
        observation_batch, info_batch = (
            self._env.reset()
        )

        obs_text = str(
            self._first(
                observation_batch
            )
        )

        info = self._unwrap_info(
            info_batch
        )

        admissible = self._as_string_list(
            info.get(
                "admissible_commands",
                [],
            )
        )

        goal_text = self._extract_goal(
            obs_text
        )

        score = float(
            info.get(
                "score",
                0.0,
            )
            or 0.0
        )

        won = bool(
            info.get(
                "won",
                False,
            )
        )

        self.state = AlfworldEnvState(
            goal_text=goal_text,
            obs_text=obs_text,
            admissible_commands=admissible,
            score=score,
            won=won,
            done=False,
            step_count=0,
        )

        return EnvResetResponse(
            observation=self.observe(),
            info={
                "won": won,
                "score": score,
                "admissible_commands": list(
                    admissible
                ),
            },
        )

    def step(
        self,
        action: Action,
    ) -> EnvResponse:
        """
        Execute one Agent action in the real TextWorld runtime.

        Important:

            DO NOT call Do.invoke() here.

        ALFWorld runtime is owned by this Bridge.
        """

        if self._env is None:
            raise RuntimeError(
                "ALFWorld environment has not been reset"
            )

        # --------------------------------------------------------
        # Validate tool name
        # --------------------------------------------------------

        if action.name != Do.name:
            return EnvResponse(
                observation=self.observe(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "error": "unknown_action"
                },
            )

        # --------------------------------------------------------
        # Validate command argument
        # --------------------------------------------------------

        text = action.kwargs.get(
            "text"
        )

        if not isinstance(text, str):
            return EnvResponse(
                observation=self.observe(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "error": "bad_args"
                },
            )

        text = text.strip()

        if not text:
            return EnvResponse(
                observation=self.observe(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "error": "empty_command"
                },
            )

        # --------------------------------------------------------
        # REAL ENVIRONMENT EXECUTION
        # --------------------------------------------------------
        #
        # ALFWorld uses a batch API.
        # Because our bridge uses batch_size=1:
        #
        #     "open fridge 1"
        #
        # becomes:
        #
        #     ["open fridge 1"]
        #
        observation_batch, score_batch, done_batch, info_batch = (
            self._env.step(
                [text]
            )
        )

        # --------------------------------------------------------
        # Remove ALFWorld batch dimension
        # --------------------------------------------------------

        obs_text = str(
            self._first(
                observation_batch
            )
        )

        score = float(
            self._first(
                score_batch
            )
            or 0.0
        )

        done = bool(
            self._first(
                done_batch
            )
        )

        info = self._unwrap_info(
            info_batch
        )

        admissible = self._as_string_list(
            info.get(
                "admissible_commands",
                [],
            )
        )

        won = bool(
            info.get(
                "won",
                False,
            )
        )

        # --------------------------------------------------------
        # Synchronize our pure state mirror
        # --------------------------------------------------------

        self.state.obs_text = obs_text
        self.state.admissible_commands = (
            admissible
        )

        self.state.score = score
        self.state.won = won
        self.state.done = done

        self.state.step_count += 1

        # --------------------------------------------------------
        # EnvHarness reward
        # --------------------------------------------------------
        #
        # Keep it simple:
        #
        #     won -> 1
        #     otherwise -> 0
        #
        # The raw ALFWorld score is still preserved in info/state.
        reward = (
            1.0
            if won
            else 0.0
        )

        return EnvResponse(
            observation=self.observe(),
            reward=reward,
            terminated=done,
            truncated=False,
            info={
                "won": won,
                "score": score,
                "admissible_commands": list(
                    admissible
                ),
            },
        )

    # ============================================================
    # Observation / evaluation
    # ============================================================

    def observe(
        self,
    ) -> Observation:
        """
        Convert pure ALFWorld state into the observation seen by Policy.

        We expose admissible commands twice:

            1. text
               -> LLM can read them directly

            2. data
               -> Rules can inspect them programmatically
        """

        if self.state.admissible_commands:
            commands_text = "\n".join(
                f"- {command}"
                for command
                in self.state.admissible_commands
            )
        else:
            commands_text = "(none)"

        parts = []

        if self.state.goal_text:
            parts.append(
                f"Task: {self.state.goal_text}"
            )

        if self.state.obs_text:
            parts.append(
                self.state.obs_text
            )

        parts.append(
            "Admissible commands:\n"
            + commands_text
        )

        return Observation(
            text="\n\n".join(parts),
            data={
                "goal_text": (
                    self.state.goal_text
                ),
                "obs_text": (
                    self.state.obs_text
                ),
                "admissible_commands": list(
                    self.state.admissible_commands
                ),
                "score": (
                    self.state.score
                ),
                "won": (
                    self.state.won
                ),
                "done": (
                    self.state.done
                ),
                "step_count": (
                    self.state.step_count
                ),
            },
        )

    def evaluate(
        self,
    ) -> EvaluationResult:
        return EvaluationResult(
            success=self.state.won,
            score=self.state.score,
            metrics={
                "steps": (
                    self.state.step_count
                ),
            },
        )

    # ============================================================
    # Rules-facing state
    # ============================================================

    def get_env_state(
        self,
    ) -> AlfworldEnvState:
        return self.state

    @classmethod
    def env_state_schema(
        cls,
    ) -> str:
        return (
            "env_state is an AlfworldEnvState with fields:\n"
            "  goal_text: str\n"
            "  obs_text: str\n"
            "  admissible_commands: list[str]\n"
            "  score: float\n"
            "  won: bool\n"
            "  done: bool\n"
            "  step_count: int\n"
            "\n"
            "The actual TextWorld runtime is private to AlfworldEnv "
            "and is not accessible through env_state."
        )

    # ============================================================
    # Runtime cleanup
    # ============================================================

    def close(
        self,
    ) -> None:
        """
        Release the real ALFWorld runtime if possible.
        """

        env = self._env

        self._env = None
        self._current_split = None

        if env is None:
            return

        close_fn = getattr(
            env,
            "close",
            None,
        )

        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                # cleanup must not mask the real episode result
                pass