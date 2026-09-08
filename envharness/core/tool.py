from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str = ""
    description: str = ""

    @classmethod
    @abstractmethod
    def invoke(cls, env_state: Any, **kwargs):
        ...