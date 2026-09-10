from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Message:
    role: str
    content: str = ""

@dataclass
class ChatResponse:
    content: str

class LLMClient(ABC):

    model_id: str = ""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None
    ) -> ChatResponse:
        ...


class ScriptedClient(LLMClient):

    def __init__(self, responses: list[str], model_id: str = "scripted"):
        self.responses = list(responses)
        self.model_id = model_id
        self.index = 0

    def reset(self):
        self.index = 0

    def chat(self, messages, *, temperature = 0.7, max_tokens = None):
        if self.index >= len(self.responses):
            raise RuntimeError(
                "scripted LLM responses exhausted"
            )

        content = self.responses[self.index]
        self.index += 1

        return ChatResponse(
            content=content
        )