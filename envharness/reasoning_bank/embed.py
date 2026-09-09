def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"embedding dim mismatch: {len(a)} vs {len(b)}"
        )

    s, na, nb = 0.0, 0.0, 0.0
    for x, y in zip(a, b):
        s += x * y
        na += x * x
        nb += y * y

    return s / max((na * nb) ** 0.5, 1e-9)

