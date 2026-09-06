"""Command line entry point: `gpu-bench-plot`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import (
    SUPPORTED_FIT_SCHEMA,
    SUPPORTED_POINTS_SCHEMA,
    SchemaMismatch,
    Sweep,
    is_application_benchmark,
    load_json,
)
from .figures import (
    draw_cases,
    draw_distribution,
    draw_fit,
    draw_heatmap,
    draw_phases,
    draw_sweep,
)
from .theme import THEMES, apply_theme

FIGURES = ("latency", "bandwidth", "fit", "heatmap", "dist", "cases", "phases")

# "sweep" was the original name. It described the method (a message-size sweep)
# rather than the quantity, which made the latency curve hard to find among
# figures otherwise named for what they show. Accepted so old commands keep
# working.
FIGURE_ALIASES = {"sweep": "latency"}

EPILOG = """\
Produce the input with benchscribe:

  python3 tools/benchscribe results --format json       > points.json
  python3 tools/benchscribe results --fit --format json > fit.json

Every figure is written alongside a companion .csv table view.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gpu-bench-plot",
        description="Figures for gpu-comm-bench results, drawn from Benchscribe JSON.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--points", type=Path, required=True, help="benchscribe --format json output"
    )
    parser.add_argument(
        "--fit",
        type=Path,
        help="benchscribe --fit --format json output (needed for the fit figure)",
    )
    parser.add_argument("--figure", choices=(*FIGURES, *FIGURE_ALIASES, "all"), default="all")
    parser.add_argument("--benchmark", help="only plot this benchmark (e.g. halo_1d)")
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--imbalance",
        type=float,
        default=None,
        help="reference line on the cases figure, e.g. moe's measured expert imbalance "
        "(3.21 for hotspot80); a bar reaching it is purely imbalance-limited",
    )
    parser.add_argument(
        "--size",
        default="min",
        help="message size in bytes for the dist figure: 'min' (default), 'max', "
        "or a byte count, snapped to the nearest swept size",
    )
    parser.add_argument(
        "--include-single-rank",
        action="store_true",
        help="keep single-rank topologies (1n1g) in the figures; they contain no "
        "communication and are excluded by default",
    )
    parser.add_argument("--theme", choices=tuple(THEMES), default="light")
    parser.add_argument("--format", dest="ext", choices=("svg", "png", "pdf"), default="svg")
    return parser.parse_args(argv)


def render_benchmark(
    payload: dict, fit_payload: dict | None, benchmark: str, args: argparse.Namespace, theme: dict
) -> tuple[list[Path], bool]:
    """Draw the requested figures for one benchmark. Returns (paths, ok)."""
    sweep = Sweep(payload.get("points", []), benchmark, args.include_single_rank)
    if not sweep.curves:
        print(f"error: no valid points for benchmark {benchmark}", file=sys.stderr)
        return [], False

    if sweep.excluded_topologies:
        print(
            f"{benchmark}: excluding single-rank {', '.join(sweep.excluded_topologies)} "
            f"(no communication); --include-single-rank to keep",
            file=sys.stderr,
        )

    wanted = FIGURES if args.figure == "all" else (args.figure,)
    written: list[Path] = []

    if "latency" in wanted:
        written.append(
            draw_sweep(sweep, theme, "latency", args.outdir, f"{benchmark}-latency", args.ext)
        )
    if "bandwidth" in wanted:
        written.append(
            draw_sweep(sweep, theme, "bandwidth", args.outdir, f"{benchmark}-bandwidth", args.ext)
        )
    if "cases" in wanted:
        out = draw_cases(sweep, theme, args.outdir, f"{benchmark}-cases", args.ext, args.imbalance)
        if out is None:
            print(f"note: {benchmark} has a single case, skipping cases figure", file=sys.stderr)
        else:
            written.append(out)
    if "dist" in wanted:
        result = draw_distribution(
            sweep, theme, args.outdir, f"{benchmark}-dist", args.ext, args.size
        )
        if result is None:
            print(
                f"warning: {benchmark}: no per-run quartiles, skipping dist figure", file=sys.stderr
            )
        else:
            out, size = result
            print(f"{benchmark}: dist figure drawn at {size} bytes", file=sys.stderr)
            written.append(out)
    if "phases" in wanted:
        result = draw_phases(sweep, theme, args.outdir, f"{benchmark}-phases", args.ext, args.size)
        if result is None:
            # Not a failure: only cg_step carries a breakdown, and only when
            # measured with GPU_BENCH_CG_PHASES=1.
            if args.figure == "phases":
                print(
                    f"warning: {benchmark}: no phase breakdown in these results; "
                    f"re-run with GPU_BENCH_CG_PHASES=1",
                    file=sys.stderr,
                )
        else:
            out, size = result
            print(f"{benchmark}: phases figure drawn at {size} bytes", file=sys.stderr)
            written.append(out)
    if "heatmap" in wanted:
        out = draw_heatmap(sweep, theme, args.outdir, f"{benchmark}-speedup", args.ext)
        if out is None:
            print(
                f"warning: {benchmark}: no baseline-relative points, skipping heatmap",
                file=sys.stderr,
            )
        else:
            written.append(out)
    if "fit" in wanted and is_application_benchmark(benchmark):
        # Not a failure: an application benchmark has one message size by
        # construction, so there is no size axis to fit. Only say so when the
        # figure was named explicitly - otherwise --figure all gets noisy.
        if args.figure == "fit":
            print(
                f"warning: {benchmark} is an application benchmark (single message "
                f"size); an alpha-beta fit is not meaningful",
                file=sys.stderr,
            )
    elif "fit" in wanted and fit_payload is not None:
        out = draw_fit(
            fit_payload.get("fits", []),
            theme,
            benchmark,
            args.outdir,
            f"{benchmark}-fit",
            args.ext,
            args.include_single_rank,
        )
        if out is None:
            print(f"warning: {benchmark}: no usable fits, skipping fit figure", file=sys.stderr)
        else:
            written.append(out)
    return written, True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.figure = FIGURE_ALIASES.get(args.figure, args.figure)
    theme = THEMES[args.theme]
    apply_theme(theme)

    try:
        payload = load_json(args.points, SUPPORTED_POINTS_SCHEMA, "points")
    except SchemaMismatch as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    points = payload.get("points", [])
    if not points:
        print(f"error: no points in {args.points}", file=sys.stderr)
        return 1

    # A results tree usually holds several benchmarks. Without this, points from
    # different benchmarks that share a (case, topology) would land on the same
    # panel while the filename claimed a single benchmark. Each gets its own set.
    available = sorted({point["benchmark"] for point in points})
    if args.benchmark:
        if args.benchmark not in available:
            print(
                f"error: no benchmark {args.benchmark!r} in {args.points}; "
                f"found: {', '.join(available)}",
                file=sys.stderr,
            )
            return 1
        benchmarks = [args.benchmark]
    else:
        benchmarks = available
        if len(benchmarks) > 1:
            print(
                f"plotting {len(benchmarks)} benchmarks: {', '.join(benchmarks)}", file=sys.stderr
            )

    fit_payload = None
    if "fit" in (FIGURES if args.figure == "all" else (args.figure,)):
        if args.fit is None:
            # Only a hard error when the fit figure was asked for by name;
            # `--figure all` should still produce everything else.
            message = "the fit figure needs --fit <benchscribe --fit --format json>"
            if args.figure == "fit":
                print(f"error: {message}", file=sys.stderr)
                return 2
            print(f"warning: skipping fit figure - {message}", file=sys.stderr)
        else:
            try:
                fit_payload = load_json(args.fit, SUPPORTED_FIT_SCHEMA, "fit")
            except SchemaMismatch as error:
                print(f"error: {error}", file=sys.stderr)
                return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failures = 0
    for benchmark in benchmarks:
        paths, ok = render_benchmark(payload, fit_payload, benchmark, args, theme)
        written.extend(paths)
        failures += 0 if ok else 1

    for path in written:
        print(path)
        print(path.with_suffix(".csv"))
    return 1 if failures and not written else 0


if __name__ == "__main__":
    raise SystemExit(main())
