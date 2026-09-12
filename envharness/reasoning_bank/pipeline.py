from envharness.core.types import Trace
from envharness.reasoning_bank.bank import Bank, MemoryItem
from envharness.reasoning_bank.embed import Embedder
from envharness.reasoning_bank.induce import LLMSkillInducer
from envharness.reasoning_bank.materialize import add_skills_to_bank

def induce_into_bank(
    *,
    task_description: str,
    traces: list[Trace],
    inducer: LLMSkillInducer,
    embedder: Embedder,
    bank: Bank,
) -> list[MemoryItem]:

    skills = inducer.induce(task_description, traces)
    return add_skills_to_bank(bank, skills, embedder)

def group_traces_by_task(traces: list[Trace]) -> dict[int | str | None, list[Trace]]:
    groups = {}
    for trace in traces:
        groups.setdefault(trace.task_id, []).append(trace)
    return groups

def build_bank_from_corpus(
    *,
    traces: list[Trace],
    task_descriptions: dict[int | str, str],
    inducer: LLMSkillInducer,
    embedder: Embedder,
    bank: Bank | None = None,
) -> Bank:
    if bank is None:
        bank = Bank()

    groups = group_traces_by_task(traces)

    for task_id, task_traces in groups.items():
        if task_id is None:
            continue

        task_description = (task_descriptions.get(task_id, str(task_id)))

        induce_into_bank(
            task_description=task_description,
            traces=task_traces,
            inducer=inducer,
            embedder=embedder,
            bank=bank,
        )

    return bank