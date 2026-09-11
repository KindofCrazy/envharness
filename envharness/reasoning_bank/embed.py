from abc import ABC, abstractmethod

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

class Embedder(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

class ScriptedEmbedder(Embedder):

    def __init__(
        self,
        embeddings: list[list[float]],
    ):
        self.embeddings = [
            list(v)
            for v in embeddings
        ]
        self.index = 0

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        end = self.index + len(texts)

        if end > len(self.embeddings):
            raise RuntimeError(
                "scripted embeddings exhausted"
            )

        result = self.embeddings[
            self.index:end
        ]

        self.index = end

        return [
            list(v)
            for v in result
        ]
