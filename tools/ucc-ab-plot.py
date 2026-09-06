#!/usr/bin/env python3
"""Plot a UCC on/off A/B produced by tools/ucc-ab.py.

The A/B is a ratio of two campaigns, which none of the gpu-bench-plot figures
draw: they all take one points.json and show absolute values. Rather than teach
that tool's CLI to accept two inputs, this reads the Markdown table ucc-ab.py
already emits and borrows the package's theme and chrome, so a backend's colour
here is the colour it has in every other figure.

Laid out like draw_sweep -- small-multipled by topology, backends as the series,
one shared figure legend -- because the claim is read across the panels: the
gain appears only once the communicator is large enough for Open MPI to select
modified_bruck, and the backends that cannot route collectives through Open MPI
stay flat in every panel.

Usage:
  tools/plot/.venv/bin/python tools/ucc-ab-plot.py UCC-AB-alltoall.md -o outdir
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "plot" / "src"))

from matplotlib.lines import Line2D  # noqa: E402

from gpu_bench_plot._mpl import plt  # noqa: E402
from gpu_bench_plot.data import (  # noqa: E402
    format_bytes,
    is_single_rank,
    topology_key,
    topology_ranks,
    write_table,
)
from gpu_bench_plot.figures import panel_grid  # noqa: E402
from gpu_bench_plot.theme import (  # noqa: E402
    BACKEND_ORDER,
    THEMES,
    apply_theme,
    colour_for,
    figure_legend,
    style_axes,
)

# Backends whose collectives reach Open MPI's coll framework, and so can see UCC
# at all. Everything else is a control: it shares the fabric and the allocation
# but cannot respond to the variable, so a flat line is the expected result and
# a moving one would mean the A/B is measuring something else.
TREATMENT = ("cuda_mpi", "sycl_mpi")

# A speedup axis wants speedup labels; log decades would read "10^1".
RATIO_TICKS = (0.5, 1, 2, 10, 30)

# Open MPI selects modified_bruck below this per-peer size, and only once the
# communicator reaches 16 ranks. Drawn only in the panels where both hold, and
# only for alltoall: the thresholds in ompi_coll_tuned_alltoall_intra_dec_fixed
# say nothing about any other collective, and allreduce has its own decision
# function with its own constants.
BRUCK_MAX_BYTES = 512
BRUCK_MIN_RANKS = 16
BRUCK_BENCHMARK = "alltoall"

# alltoall sends its count to every peer; allreduce's count is the whole message.
SIZE_LABELS = {"alltoall": "bytes per peer"}
DEFAULT_SIZE_LABEL = "message size (bytes)"

# The benchmark sends `n` float32 elements to each peer.
ELEMENT_BYTES = 4

ROW = re.compile(r"\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)x")
HEADING = re.compile(r"##\s+(\S+)\s*/\s*(\S+)")


def read_ab(path: Path) -> list[dict]:
    """Rows tagged with their benchmark.

    A joined table can hold more than one: ucc-ab.py emits a section per
    (benchmark, topology), and a CSV pair covering two benchmarks produces both.
    Rows are keyed on (topology, backend, n), which collides across benchmarks,
    so the benchmark has to travel with the row or two curves silently overlay.
    """
    rows: list[dict] = []
    benchmark, topology = "", None
    for line in path.read_text().splitlines():
        heading = HEADING.match(line)
        if heading:
            benchmark, topology = heading.group(1), heading.group(2)
            continue
        row = ROW.match(line)
        if row and topology:
            on, off = float(row.group(3)), float(row.group(4))
            rows.append(
                {
                    "benchmark": benchmark,
                    "topology": topology,
                    "n": int(row.group(1)),
                    "backend": row.group(2),
                    "bytes": int(row.group(1)) * ELEMENT_BYTES,
                    "on": on,
                    "off": off,
                    # Recomputed rather than read from the table's `ratio`, which
                    # rounds to 2 dp -- visible at the top of an axis spanning
                    # 0.5x to 30x.
                    "speedup": off / on,
                }
            )
    if not rows:
        raise SystemExit(f"no A/B rows found in {path}")
    return rows


def backends_in(rows: list[dict]) -> list[str]:
    """Fixed palette order first, so colour follows the backend not the subset."""
    present = {row["backend"] for row in rows}
    known = [b for b in BACKEND_ORDER if b in present]
    return known + sorted(present - set(known))


def draw_bars(
    benchmark: str, rows: list[dict], theme: dict, outdir: Path, stem: str, ext: str
) -> Path:
    """A panel per backend x topology, one bar per message size, anchored at 1.0.

    Bars beat lines here because the quantity is a deviation from parity, not a
    trajectory: anchored at 1.0 a gain grows up and a loss grows down, and the
    three control backends read as empty rows rather than as an unresolvable
    tangle along y=1 -- which is what the line version made of them.
    """
    apply_theme(theme)
    topologies = sorted({row["topology"] for row in rows}, key=topology_key)
    backends = backends_in(rows)
    sizes = sorted({row["bytes"] for row in rows})
    index = {size: i for i, size in enumerate(sizes)}

    # Panels grow as rows are dropped: two backends want more height each than
    # five do, or the bars flatten into a strip.
    fig, axes = panel_grid(
        len(backends),
        len(topologies),
        width=3.0,
        height=1.75 if len(backends) > 3 else 2.5,
        sharex=True,
        sharey=True,
    )
    table: list[list] = []
    marked_threshold = False

    for row_i, backend in enumerate(backends):
        control = backend not in TREATMENT
        colour = colour_for(backend, theme)
        for col, topology in enumerate(topologies):
            ax = axes[row_i][col]
            style_axes(ax, theme)
            # Bars already mark the size categories; vertical rules only add ink.
            ax.grid(False)
            ax.grid(True, axis="y", alpha=0.55)
            ax.set_axisbelow(True)
            series = sorted(
                (r for r in rows if r["backend"] == backend and r["topology"] == topology),
                key=lambda r: r["bytes"],
            )
            if series:
                ax.bar(
                    [index[point["bytes"]] for point in series],
                    [point["speedup"] - 1.0 for point in series],
                    bottom=1.0,
                    width=0.8,
                    color=colour,
                    edgecolor=theme["surface"],
                    linewidth=0.4,
                    # Controls are the evidence that the effect is real, so they
                    # must be visible as near-zero bars rather than absent ink.
                    alpha=0.55 if control else 1.0,
                )
                for point in series:
                    table.append(
                        [topology, backend, point["bytes"], point["n"], point["on"],
                         point["off"], round(point["speedup"], 4)]
                    )

            # On a 0.5x-30x log scale a control flat to within a few percent
            # is indistinguishable from an empty panel, which reads as missing
            # data rather than as the null result it is. State the worst
            # deviation so the absence of bars is quantified.
            if control and series:
                worst = max(series, key=lambda point: abs(point["speedup"] - 1.0))
                ax.annotate(
                    f"max {abs(worst['speedup'] - 1.0) * 100:.1f}%",
                    xy=(0.97, 0.88),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=7.5,
                    color=theme["muted"],
                )

            ax.axhline(1.0, color=theme["muted"], linewidth=0.9, zorder=3)
            ranks = topology_ranks(topology)
            if (
                benchmark == BRUCK_BENCHMARK
                and ranks
                and ranks >= BRUCK_MIN_RANKS
                and BRUCK_MAX_BYTES in index
            ):
                ax.axvline(
                    index[BRUCK_MAX_BYTES] - 0.5,
                    color=theme["muted"],
                    linewidth=0.9,
                    linestyle=":",
                    zorder=3,
                )
                marked_threshold = True

            ax.set_yscale("log")
            ax.set_yticks(RATIO_TICKS)
            ax.set_yticklabels([f"{tick:g}x" for tick in RATIO_TICKS])
            ax.minorticks_off()
            ax.set_xlim(-0.8, len(sizes) - 0.2)
            if row_i == 0:
                ax.set_title(f"{topology} · {ranks} ranks" if ranks else topology)
            if col == 0:
                ax.set_ylabel(f"{backend}\n(control)" if control else backend, fontsize=8.5)
            if row_i == len(backends) - 1:
                ticks = [i for i in range(len(sizes)) if i % 5 == 0]
                ax.set_xticks(ticks)
                ax.set_xticklabels([format_bytes(sizes[i]) for i in ticks], fontsize=7.5)
                ax.set_xlabel(SIZE_LABELS.get(benchmark, DEFAULT_SIZE_LABEL))

    handles, labels = [], []
    if marked_threshold:
        handles.append(Line2D([], [], color=theme["muted"], linewidth=0.9, linestyle=":"))
        labels.append(
            f"{BRUCK_MAX_BYTES} B: below this, and at >= {BRUCK_MIN_RANKS} ranks, "
            "Open MPI selects modified_bruck"
        )
        figure_legend(fig, handles, labels, theme)
    fig.suptitle(
        f"{benchmark}: speedup from enabling UCC over Open MPI's tuned component",
        color=theme["text"],
        fontsize=12,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    write_table(
        outdir / f"{stem}.csv",
        ["topology", "backend", "bytes", "n", "ucc_on_usec", "ucc_off_usec", "speedup"],
        table,
    )
    return out


def draw(benchmark: str, rows: list[dict], theme: dict, outdir: Path, stem: str, ext: str) -> Path:
    apply_theme(theme)
    topologies = sorted({row["topology"] for row in rows}, key=topology_key)
    backends = backends_in(rows)

    # One shared scale: the panels exist to be compared, and the whole point is
    # that the same backend spans 30x in one panel and 2x in another.
    # panel_grid's 3.5 x 2.9 default is sized for the multi-row sweeps; a
    # single row of it letterboxes at nearly 5:1. Taller and slightly narrower
    # keeps the 30x-to-0.5x span readable.
    fig, axes = panel_grid(
        1, len(topologies), width=3.2, height=3.9, sharex=True, sharey=True
    )
    handles: dict[str, object] = {}
    table: list[list] = []
    marked_threshold = False

    for col, topology in enumerate(topologies):
        ax = axes[0][col]
        style_axes(ax, theme)
        ranks = topology_ranks(topology)

        for backend in backends:
            series = sorted(
                (r for r in rows if r["backend"] == backend and r["topology"] == topology),
                key=lambda r: r["bytes"],
            )
            if not series:
                continue
            control = backend not in TREATMENT
            (line,) = ax.plot(
                [point["bytes"] for point in series],
                [point["speedup"] for point in series],
                color=colour_for(backend, theme),
                linewidth=2.0,
                linestyle=(0, (4, 2)) if control else "-",
                marker="s" if control else "o",
                markersize=4,
                markeredgecolor=theme["surface"],
                markeredgewidth=1.0,
                label=backend,
            )
            handles.setdefault(backend, line)
            for point in series:
                table.append(
                    [topology, backend, point["bytes"], point["n"], point["on"], point["off"],
                     round(point["speedup"], 4)]
                )

        # Parity line: above it UCC is faster, below it the tuned path is.
        ax.axhline(1.0, color=theme["muted"], linewidth=0.9, zorder=1)
        if benchmark == BRUCK_BENCHMARK and ranks and ranks >= BRUCK_MIN_RANKS:
            ax.axvline(
                BRUCK_MAX_BYTES, color=theme["muted"], linewidth=0.9, linestyle=":", zorder=1
            )
            marked_threshold = True

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_yticks(RATIO_TICKS)
        ax.set_yticklabels([f"{tick:g}x" for tick in RATIO_TICKS])
        ax.minorticks_off()
        ax.set_title(f"{topology} · {ranks} ranks" if ranks else topology)
        ax.set_xlabel(SIZE_LABELS.get(benchmark, DEFAULT_SIZE_LABEL))
        if col == 0:
            ax.set_ylabel("speedup from UCC\n(tuned / UCC)")

    ordered = [b for b in backends if b in handles]
    legend_handles = [handles[b] for b in ordered]
    legend_labels = [f"{b} (control)" if b not in TREATMENT else b for b in ordered]
    if marked_threshold:
        legend_handles.append(
            Line2D([], [], color=theme["muted"], linewidth=0.9, linestyle=":")
        )
        legend_labels.append(f"{BRUCK_MAX_BYTES} B: tuned leaves modified_bruck")
    figure_legend(fig, legend_handles, legend_labels, theme)
    fig.suptitle(
        f"{benchmark}: effect of enabling UCC over Open MPI's tuned component",
        color=theme["text"],
        fontsize=12,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    write_table(
        outdir / f"{stem}.csv",
        ["topology", "backend", "bytes", "n", "ucc_on_usec", "ucc_off_usec", "speedup"],
        table,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("table", type=Path, help="UCC-AB-*.md from tools/ucc-ab.py")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("."))
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    parser.add_argument("--ext", default="svg", help="svg (default) or png")
    parser.add_argument("--stem", default=None, help="output basename")
    parser.add_argument(
        "--include-single-rank",
        action="store_true",
        help="keep single-rank topologies (1n1g) in the figures; they contain no "
        "communication and are excluded by default",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="which benchmark to draw; required when the table holds more than one",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="comma-separated subset to draw (default: every backend in the table). "
        "Narrowing to the treatment backends drops the controls from the figure, so "
        "the null result has to be stated in the caption instead.",
    )
    parser.add_argument(
        "--kind",
        choices=("bars", "lines"),
        default="bars",
        help="bars (default): deviation from parity, one panel per backend x topology; "
        "lines: backends overlaid, one panel per topology",
    )
    args = parser.parse_args()

    rows = read_ab(args.table)
    found = sorted({row["benchmark"] for row in rows})
    if args.benchmark:
        if args.benchmark not in found:
            raise SystemExit(f"{args.table} has no {args.benchmark}; it has: {', '.join(found)}")
        benchmark = args.benchmark
    elif len(found) > 1:
        # Merging them would key rows on (topology, backend, n) alone and overlay
        # two different collectives in one panel. Refuse rather than guess.
        raise SystemExit(
            f"{args.table} holds {len(found)} benchmarks ({', '.join(found)}); "
            "pass --benchmark to choose one"
        )
    else:
        benchmark = found[0]
    rows = [row for row in rows if row["benchmark"] == benchmark]

    # A single-rank A/B has nothing to compare: both arms run the same local
    # memcpy, so the panel sits on 1.0 and reads as "UCC does nothing here"
    # rather than "there is no collective here".
    if not args.include_single_rank:
        dropped = sorted({r["topology"] for r in rows if is_single_rank(r["topology"])})
        if dropped:
            rows = [row for row in rows if not is_single_rank(row["topology"])]
            print(
                f"{benchmark}: excluding single-rank {', '.join(dropped)} "
                "(no communication); --include-single-rank to keep",
                file=sys.stderr,
            )
    if args.backends:
        wanted = {name.strip() for name in args.backends.split(",")}
        unknown = wanted - {row["backend"] for row in rows}
        if unknown:
            raise SystemExit(f"not in {args.table}: {', '.join(sorted(unknown))}")
        rows = [row for row in rows if row["backend"] in wanted]
    stem = args.stem or args.table.stem.lower()
    render = draw_bars if args.kind == "bars" else draw
    print(render(benchmark, rows, THEMES[args.theme], args.outdir, stem, args.ext))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
