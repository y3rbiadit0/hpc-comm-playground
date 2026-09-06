from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from model import (
    BASELINE_BACKEND,
    CG_PHASE_NAMES,
    METRIC_SPECS,
    AcrossRuns,
    BackendSummary,
    GroupedMeasurements,
    GroupKey,
    Measurement,
    PhaseBreakdown,
    RunStats,
    MetricName,
    MetricSpec,
    SummaryGroups,
    SummaryRow,
    Status,
)


def parse_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def quantile(sorted_values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile. Exact enough for the handful of runs we have, and
    it never invents a value that was not measured."""
    if not sorted_values:
        return None
    index = min(int(fraction * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


def run_stats(record: Measurement, metric: MetricSpec) -> RunStats:
    """One record's own numbers. The percentile fields are absent from results
    produced before the harness emitted them."""
    get = lambda key: parse_float(record.fields.get(key))
    iters = get("iters")
    return RunStats(
        job=record.job,
        mean=get(metric.name.value),
        median=get("median_usec"),
        p25=get("p25_usec"),
        p75=get("p75_usec"),
        minimum=get("min_usec"),
        maximum=get("max_usec"),
        stddev=get("stddev_usec"),
        iterations=int(iters) if iters is not None else None,
    )


def phase_breakdown(records: list[Measurement]) -> PhaseBreakdown | None:
    """Average the per-phase fields over the trials that carry them.

    Returns None when no record has them, which is the common case: the phase
    pass is opt-in, so most results have no breakdown at all. That is different
    from a breakdown whose phases are zero, and the two must not be conflated.
    """
    means: dict[str, float | None] = {}
    for name in (*CG_PHASE_NAMES, "sum"):
        values = [
            value
            for value in (
                parse_float(record.fields.get(f"phase_{name}_usec")) for record in records
            )
            if value is not None
        ]
        means[name] = mean_or_none(values)
    breakdown = PhaseBreakdown(
        pack=means["pack"],
        halo=means["halo"],
        compute=means["compute"],
        reduce=means["reduce"],
        total=means["sum"],
    )
    return None if breakdown.is_empty() else breakdown


def across_runs(records: list[Measurement], metric: MetricSpec) -> AcrossRuns:
    """Spread over independent jobs.

    Trials inside one job share an allocation, so they are averaged into a single
    value first; the dispersion is then over jobs. Treating trials as independent
    samples would understate it several-fold.
    """
    per_job: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = parse_float(record.fields.get(metric.name.value))
        if value is not None:
            per_job[record.job].append(value)
    job_values = sorted(statistics.fmean(values) for values in per_job.values() if values)
    if not job_values:
        return AcrossRuns()
    return AcrossRuns(
        n_runs=len(job_values),
        median=statistics.median(job_values),
        p25=quantile(job_values, 0.25),
        p75=quantile(job_values, 0.75),
        # Undefined for a single job, and meaningless enough below ~5 that the
        # caller should show n_runs alongside it.
        stddev=statistics.stdev(job_values) if len(job_values) > 1 else None,
    )


def preferred_metric(records: list[Measurement]) -> MetricSpec:
    records = [record for record in records if record.status == Status.OK and record.valid]
    for metric in (MetricName.USEC, MetricName.TIME_PER_ITER_S, MetricName.TIME_S):
        if any(metric.value in record.fields for record in records):
            return METRIC_SPECS[metric]
    return METRIC_SPECS[MetricName.USEC]


def metric_value_for(summary: BackendSummary, metric: MetricSpec) -> float | None:
    if metric.name == MetricName.GBYTES_PER_S:
        return summary.bandwidth
    return summary.metric_value


def metric_min_for(summary: BackendSummary, metric: MetricSpec) -> float | None:
    if metric.name == MetricName.GBYTES_PER_S:
        return None
    return summary.metric_min


def relative_delta(base: float | None, value: float | None) -> float | None:
    if base is None or value is None or base == 0:
        return None
    return (value - base) / base * 100.0


def relative_speedup(
    base: float | None, value: float | None, lower_is_better: bool
) -> float | None:
    if base is None or value is None or base == 0 or value == 0:
        return None
    if lower_is_better:
        return base / value
    return value / base


@dataclass(frozen=True)
class SummaryTable:
    baseline: str
    metric_by_benchmark: dict[str, MetricSpec]
    groups: SummaryGroups

    @classmethod
    def from_measurements(
        cls,
        measurements: list[Measurement],
        baseline: str = BASELINE_BACKEND,
        metric_override: MetricName | None = None,
    ) -> "SummaryTable":
        grouped = cls._group_measurements(measurements)
        metric_by_benchmark = cls._select_metrics(grouped, metric_override)
        groups = cls._aggregate_groups(grouped, metric_by_benchmark)
        return cls(baseline=baseline, metric_by_benchmark=metric_by_benchmark, groups=groups)

    @staticmethod
    def _group_measurements(measurements: list[Measurement]) -> GroupedMeasurements:
        grouped: GroupedMeasurements = defaultdict(list)
        for measurement in measurements:
            key = (
                measurement.benchmark,
                measurement.topology,
                measurement.n,
                measurement.case,
                measurement.backend,
            )
            grouped[key].append(measurement)
        return dict(grouped)

    @staticmethod
    def _select_metrics(
        grouped: GroupedMeasurements, metric_override: MetricName | None
    ) -> dict[str, MetricSpec]:
        if metric_override is not None:
            benchmarks = {benchmark for benchmark, _topology, _n, _case, _backend in grouped}
            return {benchmark: METRIC_SPECS[metric_override] for benchmark in benchmarks}

        records_by_benchmark: dict[str, list[Measurement]] = defaultdict(list)
        for (benchmark, _topology, _n, _case, _backend), records in grouped.items():
            records_by_benchmark[benchmark].extend(records)
        return {
            benchmark: preferred_metric(records)
            for benchmark, records in records_by_benchmark.items()
        }

    @staticmethod
    def _aggregate_groups(
        grouped: GroupedMeasurements, metric_by_benchmark: dict[str, MetricSpec]
    ) -> SummaryGroups:
        groups: SummaryGroups = defaultdict(dict)
        for (benchmark, topology, n, case, backend), records in grouped.items():
            metric = metric_by_benchmark[benchmark]
            supported_records = [
                record for record in records if record.status == Status.OK and record.valid
            ]
            metric_values = [
                value
                for value in (
                    parse_float(record.fields.get(metric.name.value))
                    for record in supported_records
                )
                if value is not None
            ]
            bandwidths = [
                value
                for value in (
                    parse_float(record.fields.get(MetricName.GBYTES_PER_S.value))
                    for record in supported_records
                )
                if value is not None and value > 0.0
            ]
            nbytes = next(
                (
                    int(value)
                    for value in (
                        parse_float(record.fields.get("bytes")) for record in supported_records
                    )
                    if value is not None
                ),
                None,
            )
            if any(
                record.status == Status.ERROR
                or (record.status != Status.NOT_IMPLEMENTED and not record.valid)
                for record in records
            ):
                status = Status.ERROR
            elif supported_records:
                status = Status.OK
            else:
                status = Status.NOT_IMPLEMENTED
            groups[GroupKey(benchmark, topology, n, case)][backend] = BackendSummary(
                backend=backend,
                metric_value=mean_or_none(metric_values),
                metric_min=min(metric_values) if metric_values else None,
                bandwidth=mean_or_none(bandwidths),
                nbytes=nbytes,
                trials=len(supported_records) if supported_records else len(records),
                valid_all=status != Status.ERROR,
                status=status,
                runs=tuple(run_stats(record, metric) for record in supported_records),
                across_runs=across_runs(supported_records, metric),
                phases=phase_breakdown(supported_records),
            )
        return {key: dict(value) for key, value in groups.items()}

    def benchmarks(self) -> list[str]:
        return sorted({key.benchmark for key in self.groups})

    def cases_for(self, benchmark: str) -> list[str]:
        return sorted({key.case for key in self.groups if key.benchmark == benchmark})

    def topologies_for(self, benchmark: str, case: str = "") -> list[str]:
        return sorted(
            {key.topology for key in self.groups if key.benchmark == benchmark and key.case == case}
        )

    def sizes_for(self, benchmark: str, topology: str, case: str = "") -> list[int]:
        return sorted(
            key.n
            for key in self.groups
            if key.benchmark == benchmark and key.topology == topology and key.case == case
        )

    def rows(self) -> list[SummaryRow]:
        rows: list[SummaryRow] = []
        for benchmark in self.benchmarks():
            for case in self.cases_for(benchmark):
                for topology in self.topologies_for(benchmark, case):
                    for n in self.sizes_for(benchmark, topology, case):
                        rows.extend(self.rows_for(benchmark, topology, n, case))
        return rows

    def rows_for(self, benchmark: str, topology: str, n: int, case: str = "") -> list[SummaryRow]:
        key = GroupKey(benchmark, topology, n, case)
        summaries = self.groups[key]
        metric = self.metric_by_benchmark[benchmark]
        base_summary = summaries.get(self.baseline)
        base_value = metric_value_for(base_summary, metric) if base_summary is not None else None
        rows = [
            self._row_from_summary(key, metric, summary, base_value)
            for summary in summaries.values()
        ]
        return sorted(rows, key=self._row_sort_key)

    def _row_from_summary(
        self,
        key: GroupKey,
        metric: MetricSpec,
        summary: BackendSummary,
        base_value: float | None,
    ) -> SummaryRow:
        value = metric_value_for(summary, metric)
        return SummaryRow(
            benchmark=key.benchmark,
            topology=key.topology,
            n=key.n,
            metric=metric,
            backend=summary.backend,
            value=value,
            value_min=metric_min_for(summary, metric),
            bandwidth=summary.bandwidth,
            nbytes=summary.nbytes,
            delta_pct_vs_base=relative_delta(base_value, value),
            speedup_vs_base=relative_speedup(base_value, value, metric.lower_is_better),
            trials=summary.trials,
            valid_all=summary.valid_all,
            case=key.case,
            status=summary.status,
            phases=summary.phases,
        )

    def _row_sort_key(self, row: SummaryRow) -> tuple[bool, float, str]:
        speedup = row.speedup_vs_base
        return (speedup is None, -(speedup or 0.0), row.backend)
