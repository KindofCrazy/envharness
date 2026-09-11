from envharness.reasoning_bank.bank import MemoryItem, Bank
from envharness.reasoning_bank.embed import Embedder
from envharness.reasoning_bank.induce import SkillDraft


def skill_embedding_text(skill: SkillDraft,) -> str:
    return (
        f"{skill.title}"
        f"{skill.description}"
    )

def materialize_skills(
    skills: list[SkillDraft],
    embedder: Embedder
) -> list[MemoryItem]:
    if not skills:
        return []

    texts = [skill_embedding_text(skill) for skill in skills]
    embeddings = embedder.embed(texts)

    if len(embeddings) != len(skills):
        raise ValueError(
            "embedding count must match "
            "skill count"
        )

    items = []

    for skill, embedding in zip(skills, embeddings):
        if not isinstance(embedding, list):
            raise ValueError(
                "embedding must be a list"
            )

        if not embedding:
            raise ValueError(
                "embedding must not be empty"
            )

        items.append(
            MemoryItem(
                title=skill.title,
                description=skill.description,
                content=skill.content,
                embedding=list(embedding),
                source=dict(skill.source),
            )
        )

    return items

def add_skills_to_bank(bank: Bank, skills: list[SkillDraft], embedder: Embedder) -> list[MemoryItem]:
    items = materialize_skills(skills, embedder)
    bank.add(items)
    return items
