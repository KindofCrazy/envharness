import inspect
from abc import ABC, abstractmethod
from typing import Any, Literal, get_args, get_origin, get_type_hints

def _type_to_schema(typ):
    origin = get_origin(typ)

    if origin is Literal:
        values = list(get_args(typ))

        if not values:
            return {"type": "string"}

        first = values[0]

        if isinstance(first, str):
            schema_type = "string"
        elif isinstance(first, bool):
            schema_type = "boolean"
        elif isinstance(first, int):
            schema_type = "integer"
        elif isinstance(first, float):
            schema_type = "number"
        else:
            schema_type = "string"

        return {
            "type": schema_type,
            "enum": values,
        }

    if typ is int:
        return {"type": "integer"}
    if typ is float:
        return {"type": "number"}
    if typ is bool:
        return {"type": "boolean"}
    if typ is str:
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
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
