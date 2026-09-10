import inspect
from abc import ABC, abstractmethod
from typing import Any, get_type_hints

def _type_to_schema(self, typ):
    if typ is int:
        return {"type": "interger"}
    if typ is float:
        return {"type": "number"}
    if typ is bool:
        return {"type": "boolean"}
    if type is str:
        return {"type": "string"}
    return {"type": "string"}


class Tool(ABC):
    name: str = ""
    description: str = ""

    @classmethod
    @abstractmethod
    def invoke(cls, env_state: Any, **kwargs) -> Any:
        ...

    @classmethod
    def get_info(cls) -> dict:
        sig = inspect.signature(cls.invoke)
        hints = get_type_hints(cls.invoke)

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name in ("cls", "env_state"):
                continue

            typ = hints.get(name, str)

            properties[name] = _type_to_schema(typ)

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": cls.name or cls.__name__,
                "description": cls.description,
                "parameters": {
                    "type": object,
                    "properties": properties,
                    "required": required
                }
            }
        }
