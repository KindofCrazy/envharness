import json
import urllib.request
import urllib.error

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

def _message_to_dict(message) -> dict:
    return {
        "role": message.role,
        "content": message.content
    }

class OpenAICompatibleClient(LLMClient):

    def __init__(self, model_id: str, base_url: str, api_key: str | None = None, timeout: float = 60.0):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None
    ) -> ChatResponse:
        payload = {
            "model": self.model_id,
            "messages": [
                _message_to_dict(message) for message in messages
            ],
            "temperature": temperature
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        request = urllib.request.Request(
            url=(
                self.base_url + "/chat/completions"
            ),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                body = response.read().decode(
                    "utf-8"
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"LLM HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"LLM connection failed: {exc}"
            ) from exc

        try:
            data = json.loads(body)

            content = (
                data["choices"][0]
                ["message"]["content"]
            )
        except (
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "invalid chat completion response"
            ) from exc

        if not isinstance(content, str):
            raise RuntimeError(
                "chat completion content "
                "must be a string"
            )

        return ChatResponse(
            content=content
        )
