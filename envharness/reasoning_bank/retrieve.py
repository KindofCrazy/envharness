from envharness.reasoning_bank.bank import Bank, MemoryItem
from envharness.reasoning_bank.embed import Embedder

class MemoryRetriever:

    def __init__(self, bank: Bank, embedder: Embedder, *, k: int = 3, cosine_threshold: float = 0.0):
        self.bank = bank
        self.embedder = embedder
        self.k = k
        self.cosine_threshold = cosine_threshold

    def retrieve(self, query: str) -> list[MemoryItem]:
        if not self.bank.items:
            return []
        embeddings = self.embedder.embed([query])

        if len(embeddings) != 1:
            raise ValueError("query embedding must contain exactly one vector")

        query_embedding = embeddings[0]

        if not query_embedding:
            raise ValueError("query embedding must not be empty")

        return self.bank.retrieve(
            query_embedding=query_embedding,
            k=self.k,
            cosine_threshold=self.cosine_threshold,
        )


def render_memories(memories: list[MemoryItem]) -> str:
    if not memories:
        return ""

    return "\n\n".join(memory.text for memory in memories)