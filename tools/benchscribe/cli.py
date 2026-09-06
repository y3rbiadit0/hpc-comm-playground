from __future__ import annotations

import argparse
import sys
from pathlib import Path

from characterize import characterize
from model import MetricName, OutputFormat
from render import (
    phase_rows,
    render_csv,
    render_fit_csv,
    render_fit_json,
    render_fit_markdown,
    render_json,
    render_markdown,
    render_phase_csv,
    render_phase_json,
    render_phase_markdown,
)
from scan import scan_results
from summary import SummaryTable


DESCRIPTION = """Summarize gpu-comm-bench benchmark results.

Benchscribe scans Slurm stdout files under a results directory, parses benchmark
key=value report lines, aggregates repeated trials, and reports every backend
relative to the cuda_mpi baseline.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results_dir", nargs="?", default="results", help="results directory (default: results)"
    )
    parser.add_argument(
        "--format",
        choices=[item.value for item in OutputFormat],
        default=OutputFormat.MARKDOWN.value,
    )
    parser.add_argument("--benchmark", help="only summarize this benchmark (e.g. allreduce)")
    parser.add_argument(
        "--metric", choices=[item.value for item in MetricName], help="override the primary metric"
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="report per-backend latency floor (α), peak bandwidth (B∞), and n½ = α·B∞ across the sweep",
    )
    parser.add_argument(
        "--phases",
        action="store_true",
        help="report cg_step's per-phase breakdown and the overlap it implies "
        "(needs results measured with GPU_BENCH_CG_PHASES=1)",
    )
    args = parser.parse_args(argv)
    if args.fit and args.phases:
        parser.error("--fit and --phases report different views; pick one")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"error: no such directory: {results_dir}", file=sys.stderr)
        return 2

    measurements = scan_results(results_dir)
    if args.benchmark:
        measurements = [
            measurement for measurement in measurements if measurement.benchmark == args.benchmark
        ]
    if not measurements:
        print(f"error: no benchmark lines found under {results_dir}", file=sys.stderr)
        return 1

    metric_override = MetricName(args.metric) if args.metric else None
    table = SummaryTable.from_measurements(measurements, metric_override=metric_override)
    if args.phases:
        if not phase_rows(table):
            print(
                "error: no result carries a phase breakdown. Re-run with "
                "GPU_BENCH_CG_PHASES=1 to record one.",
                file=sys.stderr,
            )
            return 1
        if args.format == OutputFormat.JSON.value:
            render_phase_json(table, sys.stdout)
        elif args.format == OutputFormat.CSV.value:
            render_phase_csv(table, sys.stdout)
        else:
            render_phase_markdown(table, sys.stdout)
        return 0
    if args.fit:
        chars = characterize(table)
        if args.format == OutputFormat.JSON.value:
            render_fit_json(chars, sys.stdout)
        elif args.format == OutputFormat.CSV.value:
            render_fit_csv(chars, sys.stdout)
        else:
            render_fit_markdown(chars, sys.stdout)
        return 0
    if args.format == OutputFormat.JSON.value:
        render_json(table, sys.stdout)
    elif args.format == OutputFormat.CSV.value:
        render_csv(table, sys.stdout)
    else:
        render_markdown(table, sys.stdout)
    return 0
