from __future__ import annotations

import csv
import json
from dataclasses import asdict
import datetime as dt
from typing import TextIO

from characterize import ALPHA_MAX_BYTES, Characterization
from model import CG_PHASE_NAMES, MetricName, Status, SummaryRow
from summary import SummaryTable


def format_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def format_bytes(value: float | None) -> str:
    if value is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def format_delta(row: SummaryRow, baseline: str) -> str:
    if row.backend == baseline and row.delta_pct_vs_base is not None:
        return "0.0%"
    if row.delta_pct_vs_base is None:
        return "-"
    return f"{row.delta_pct_vs_base:+.1f}%"


def format_speedup(row: SummaryRow, baseline: str) -> str:
    if row.backend == baseline and row.speedup_vs_base is not None:
        return "1.00x"
    if row.speedup_vs_base is None:
        return "-"
    return f"{row.speedup_vs_base:.2f}x"


def format_row_number(row: SummaryRow, value: float | None) -> str:
    return "N/I" if row.status == Status.NOT_IMPLEMENTED else format_number(value)


def format_row_status(row: SummaryRow) -> str:
    if row.status == Status.NOT_IMPLEMENTED:
        return "N/I"
    return "PASS" if row.valid_all else "FAIL"


def render_markdown(table: SummaryTable, out: TextIO) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out.write("# Benchmark Results Summary\n\n")
    out.write(f"_Generated {stamp}. Baseline: `{table.baseline}`._\n\n")
    out.write(
        "Latency metrics are lower-is-better; **Speedup** = baseline / backend "
        "for latency metrics and backend / baseline for bandwidth metrics. "
        "**Delta** is relative to baseline; negative latency deltas are faster.\n\n"
    )

    for benchmark in table.benchmarks():
        metric = table.metric_by_benchmark[benchmark]
        out.write(f"## {benchmark}\n\n")
        out.write(f"Primary metric: `{metric.name.value}` ({metric.unit}).\n\n")
        cases = table.cases_for(benchmark)
        show_case = cases != [""]
        for case in cases:
            if show_case:
                out.write(f"### Case: {case or '(default)'}\n\n")
            for topology in table.topologies_for(benchmark, case):
                out.write(f"{'####' if show_case else '###'} {topology}\n\n")
                out.write(
                    f"| Size (n) | Bytes | Backend | {metric.name.value} ({metric.unit}) | "
                    f"Min ({metric.unit}) | GB/s | Delta vs base | Speedup | Trials | Valid |\n"
                )
                out.write(
                    "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |\n"
                )
                for n in table.sizes_for(benchmark, topology, case):
                    for row in table.rows_for(benchmark, topology, n, case):
                        bytes_field = "-" if row.nbytes is None else str(row.nbytes)
                        delta = (
                            "N/I"
                            if row.status == Status.NOT_IMPLEMENTED
                            else format_delta(row, table.baseline)
                        )
                        speedup = (
                            "N/I"
                            if row.status == Status.NOT_IMPLEMENTED
                            else format_speedup(row, table.baseline)
                        )
                        out.write(
                            f"| {row.n} | {bytes_field} | `{row.backend}` | {format_row_number(row, row.value)} "
                            f"| {format_row_number(row, row.value_min)} | {format_row_number(row, row.bandwidth)} "
                            f"| {delta} | {speedup} | {row.trials} | {format_row_status(row)} |\n"
                        )
                out.write("\n")


def render_fit_markdown(chars: list[Characterization], out: TextIO) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out.write("# Latency / Bandwidth Characterization\n\n")
    out.write(f"_Generated {stamp}._\n\n")
    out.write(
        "α = median latency for messages up to 4 KiB. "
        "B∞ = peak bandwidth. n½ = α·B∞ (message at half of peak). "
        "Tail = bandwidth at the largest message.\n\n"
    )
    by_bench: dict[str, list[Characterization]] = {}
    for char in chars:
        by_bench.setdefault(char.benchmark, []).append(char)
    for benchmark in sorted(by_bench):
        out.write(f"## {benchmark}\n\n")
        show_case = any(char.case for char in by_bench[benchmark])
        case_header = " Case |" if show_case else ""
        case_rule = " --- |" if show_case else ""
        out.write(
            f"|{case_header} Topology | Backend | α | B∞ (GB/s) | peak @ | n½ | tail (GB/s) | pts |\n"
        )
        out.write(f"|{case_rule} --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for char in sorted(
            by_bench[benchmark], key=lambda item: (item.case, item.topology, item.backend)
        ):
            case_field = f" {char.case or '(default)'} |" if show_case else ""
            out.write(
                f"|{case_field} {char.topology} | `{char.backend}` | {format_number(char.alpha)} {char.unit} "
                f"| {format_number(char.binf_gbs)} | {format_bytes(char.peak_bytes)} "
                f"| {format_bytes(char.nhalf_bytes)} | {format_number(char.tail_gbs)} | {char.points} |\n"
            )
        out.write("\n")


def render_fit_csv(chars: list[Characterization], out: TextIO) -> None:
    writer = csv.writer(out)
    writer.writerow(
        [
            "benchmark",
            "case",
            "topology",
            "backend",
            "alpha",
            "alpha_unit",
            "binf_gbytes_per_s",
            "peak_bytes",
            "nhalf_bytes",
            "tail_gbytes_per_s",
            "points",
        ]
    )
    for char in chars:
        writer.writerow(
            [
                char.benchmark,
                char.case,
                char.topology,
                char.backend,
                "" if char.alpha is None else f"{char.alpha:.9g}",
                char.unit,
                "" if char.binf_gbs is None else f"{char.binf_gbs:.9g}",
                "" if char.peak_bytes is None else char.peak_bytes,
                "" if char.nhalf_bytes is None else f"{char.nhalf_bytes:.9g}",
                "" if char.tail_gbs is None else f"{char.tail_gbs:.9g}",
                char.points,
            ]
        )


def render_fit_json(chars: list[Characterization], out: TextIO) -> None:
    """Machine-readable α-β characterization, for the plotting tools.

    Same numbers as render_fit_csv, but typed and versioned so a figure script
    does not have to re-parse CSV or re-implement the fit. `schema_version` is
    the contract - bump it on any incompatible change.
    """
    json.dump(
        {
            "schema_version": 1,
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "alpha_max_bytes": ALPHA_MAX_BYTES,
            "fits": [asdict(char) for char in chars],
        },
        out,
        indent=2,
    )
    out.write("\n")


def render_json(table: SummaryTable, out: TextIO) -> None:
    """Machine-readable export carrying the full distribution.

    Unlike the Markdown and CSV views, which collapse each point to a single
    number, this keeps every run so figures can show spread: `runs` for the
    within-run iteration distribution (box plots), `across_runs` for the spread
    over independent jobs (error bands on a line graph).

    `schema_version` is the contract - bump it on any incompatible change.
    """
    points = []
    for key, backends in sorted(table.groups.items()):
        metric = table.metric_by_benchmark[key.benchmark]
        for backend, summary in sorted(backends.items()):
            points.append(
                {
                    "benchmark": key.benchmark,
                    "topology": key.topology,
                    "case": key.case,
                    "backend": backend,
                    "n": key.n,
                    "bytes": summary.nbytes,
                    "metric": metric.name.value,
                    "unit": metric.unit,
                    "value_mean": summary.metric_value,
                    "value_min": summary.metric_min,
                    "gbytes_per_s": summary.bandwidth,
                    "valid": summary.valid_all,
                    "status": summary.status.value,
                    "runs": [asdict(run) for run in summary.runs],
                    "across_runs": asdict(summary.across_runs) if summary.across_runs else None,
                    # Additive and optional: null wherever the phase pass did not
                    # run, which is most results. Readers that ignore it are
                    # unaffected, so this does not bump schema_version.
                    "phases": asdict(summary.phases) if summary.phases else None,
                }
            )
    json.dump(
        {
            "schema_version": 1,
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "points": points,
        },
        out,
        indent=2,
    )
    out.write("\n")


def phase_rows(table: SummaryTable) -> list[SummaryRow]:
    """Rows that carry a phase breakdown, in report order."""
    return [row for row in table.rows() if row.phases is not None]


def phase_overlap(row: SummaryRow) -> float | None:
    """How much the unsplit step overlaps its phases, in the metric's unit.

    The phase pass synchronizes between phases, so its total is the no-overlap
    cost; the reported value is what the unsplit loop achieved. Only meaningful
    when the reported metric is the per-iteration time in microseconds - against
    a bandwidth the subtraction has no meaning.
    """
    if row.metric.name != MetricName.USEC or row.value is None:
        return None
    total = row.phases.total if row.phases else None
    return None if total is None else total - row.value


def render_phase_markdown(table: SummaryTable, out: TextIO) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out.write("# Phase Breakdown\n\n")
    out.write(f"_Generated {stamp}._\n\n")
    out.write(
        "Per-phase means from the opt-in second pass (`GPU_BENCH_CG_PHASES=1`), which "
        "synchronizes between phases. **Sum** is what the step would cost with no "
        "overlap; **Reported** is the unsplit loop's per-iteration time; **Overlap** is "
        "the difference, so it measures how much the phases overlap in the loop that is "
        "actually reported.\n\n"
    )
    rows = phase_rows(table)
    if not rows:
        out.write("_No result carries a phase breakdown._\n")
        return

    current = None
    for row in rows:
        heading = (row.benchmark, row.case, row.topology)
        if heading != current:
            current = heading
            label = f"{row.benchmark} / {row.topology}"
            if row.case:
                label += f" / {row.case}"
            out.write(f"## {label}\n\n")
            out.write("| Size (n) | Backend | " + " | ".join(
                f"{name} (us)" for name in CG_PHASE_NAMES
            ) + " | Sum (us) | Reported (us) | Overlap (us) |\n")
            out.write("| ---: | --- | " + " | ".join("---:" for _ in CG_PHASE_NAMES)
                      + " | ---: | ---: | ---: |\n")
        cells = " | ".join(format_number(row.phases.phase(name)) for name in CG_PHASE_NAMES)
        out.write(
            f"| {row.n} | `{row.backend}` | {cells} | {format_number(row.phases.total)} "
            f"| {format_number(row.value)} | {format_number(phase_overlap(row))} |\n"
        )
    out.write("\n")


def phase_record(row: SummaryRow) -> dict:
    record = {
        "benchmark": row.benchmark,
        "case": row.case,
        "topology": row.topology,
        "backend": row.backend,
        "n": row.n,
        "unit": "us",
    }
    for name in CG_PHASE_NAMES:
        record[name] = row.phases.phase(name)
    record["sum"] = row.phases.total
    record["reported"] = row.value
    record["overlap"] = phase_overlap(row)
    return record


def render_phase_csv(table: SummaryTable, out: TextIO) -> None:
    fields = ["benchmark", "case", "topology", "backend", "n", "unit",
              *CG_PHASE_NAMES, "sum", "reported", "overlap"]
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for row in phase_rows(table):
        record = phase_record(row)
        writer.writerow(
            {k: ("" if v is None else f"{v:.9g}" if isinstance(v, float) else v)
             for k, v in record.items()}
        )


def render_phase_json(table: SummaryTable, out: TextIO) -> None:
    json.dump(
        {
            "schema_version": 1,
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "phases": [phase_record(row) for row in phase_rows(table)],
        },
        out,
        indent=2,
    )
    out.write("\n")


def render_csv(table: SummaryTable, out: TextIO) -> None:
    writer = csv.writer(out)
    writer.writerow(
        [
            "benchmark",
            "case",
            "topology",
            "n",
            "backend",
            "metric",
            "unit",
            "value_mean",
            "value_min",
            "gbytes_per_s",
            "delta_pct_vs_base",
            "speedup_vs_base",
            "trials",
            "valid",
            "status",
        ]
    )
    for row in table.rows():
        writer.writerow(
            [
                row.benchmark,
                row.case,
                row.topology,
                row.n,
                row.backend,
                row.metric.name.value,
                row.metric.unit,
                "" if row.value is None else f"{row.value:.9g}",
                "" if row.value_min is None else f"{row.value_min:.9g}",
                "" if row.bandwidth is None else f"{row.bandwidth:.9g}",
                "" if row.delta_pct_vs_base is None else f"{row.delta_pct_vs_base:.3f}",
                "" if row.speedup_vs_base is None else f"{row.speedup_vs_base:.4f}",
                row.trials,
                format_row_status(row),
                row.status.value,
            ]
        )
