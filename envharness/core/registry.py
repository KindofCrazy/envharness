from typing import Any

_ENV_REGISRTY: dict[str, type[Any]] = {}
_HARNESS_REGISTRY: dict[str, type[Any]] = {}

def register_env(tag: str):
    def decorator(cls):
        if tag in _ENV_REGISRTY:
            raise ValueError(f"env type already registered: {tag}")

        _ENV_REGISRTY[tag] = cls

        cls.env_type = classmethod(lambda _cls: tag)
        return cls
    return decorator

def register_harness(tag: str):
    def decorator(cls):
        if tag in _HARNESS_REGISTRY:
            raise ValueError(
                f"harness type already registered: {tag}"
            )

        _HARNESS_REGISTRY[tag] = cls

        cls.harness_type = classmethod(
            lambda _cls: tag
        )
        return cls
    return decorator

def get_env_class(tag: str):
    if tag not in _ENV_REGISRTY:
        raise KeyError(f"unkown env type: {tag}")
    return _ENV_REGISRTY[tag]

def get_harness_class(tag: str):
    if tag not in _HARNESS_REGISTRY:
        raise KeyError(f"unknown harness type: {tag}")
    return _HARNESS_REGISTRY[tag]
