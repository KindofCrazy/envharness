from envharness.core.types import ValidationBatch, BaselineRolloutSummary, BaselineSnapshot

def summarize(batch: ValidationBatch) -> BaselineSnapshot:
    summaries = [BaselineRolloutSummary(
        success=trace.success,
        steps=len(trace.steps)
    ) for trace in batch.traces]
    success_steps = [len(trace.steps) for trace in batch.traces if trace.success]
    success_rate = batch.success_count / len(batch.traces) if len(batch.traces) > 0 else 0.0

    return BaselineSnapshot(
        n=len(batch.traces),
        n_success=batch.success_count,
        success_rate=success_rate,
        avg_success_steps=sum(success_steps) / len(success_steps) if success_steps else None,
        per_rollout=summaries,
    )