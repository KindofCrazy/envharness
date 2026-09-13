import importlib


def import_symbol(path: str):
    module_name, sep, symbol_name = path.partition(":")

    if not sep:
        raise ValueError("import path must have form 'module:Symbol'")
    if not module_name or not symbol_name:
        raise ValueError("invalid import path")

    module = importlib.import_module(module_name)

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(
            f"{symbol_name!r} not found "
            f"in {module_name!r}"
        ) from exc