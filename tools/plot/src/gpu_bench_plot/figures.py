"""The figures themselves."""

from __future__ import annotations

import math
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from ._mpl import plt
from .data import Sweep, format_bytes, is_single_rank, topology_key, write_table
from .theme import (
    BACKEND_ORDER,
    BASELINE_BACKEND,
    PHASE_ORDER,
    colour_for,
    figure_legend,
    phase_colour,
    style_axes,
)


def panel_title(case: str, topology: str) -> str:
    """Benchmarks without a `case=` field carry case="" - no dangling separator."""
    return f"{case} · {topology}" if case else topology


def axis_label(case: str, measure: str) -> str:
    return f"{case}\n{measure}" if case else measure


def panel_grid(
    rows: int,
    cols: int,
    width: float = 3.5,
    height: float = 2.9,
    sharex: bool | str = False,
    sharey: bool | str = False,
):
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(width * cols, height * rows),
        squeeze=False,
        constrained_layout=True,
        sharex=sharex,
        sharey=sharey,
    )
    return fig, axes


def draw_sweep(
    sweep: Sweep, theme: dict, value_axis: str, outdir: Path, stem: str, ext: str
) -> Path:
    """Latency-or-bandwidth against message size, small-multipled by case x topology."""
    cases, topologies = sweep.cases, sweep.topologies
    # Latency is logarithmic, so one shared scale keeps every panel comparable.
    # Bandwidth is linear and intra-node runs an order of magnitude above
    # inter-node: one shared scale would flatten the inter-node panels to a
    # line. Share down each column instead, so isolated and steady stay
    # comparable within a topology and each topology keeps a readable range.
    fig, axes = panel_grid(
        len(cases),
        len(topologies),
        sharex=True,
        sharey=True if value_axis == "latency" else "col",
    )
    handles: dict[str, object] = {}
    rows: list[list] = []

    for row, case in enumerate(cases):
        for col, topology in enumerate(topologies):
            ax = axes[row][col]
            style_axes(ax, theme)
            for backend in sweep.backends():
                series = sweep.curves.get((case, topology), {}).get(backend)
                if not series:
                    continue
                colour = colour_for(backend, theme)
                xs = [point["bytes"] for point in series]
                if value_axis == "latency":
                    ys = [point["value_mean"] for point in series]
                else:
                    ys = [point["gbytes_per_s"] for point in series]
                if any(y is None for y in ys):
                    continue
                (line,) = ax.plot(
                    xs,
                    ys,
                    color=colour,
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                    markeredgecolor=theme["surface"],
                    markeredgewidth=1.0,
                    label=backend,
                )
                handles.setdefault(backend, line)

                if value_axis == "latency":
                    bands = [sweep.band(point) for point in series]
                    if all(band is not None for band in bands):
                        ax.fill_between(
                            xs,
                            [b[0] for b in bands],
                            [b[1] for b in bands],
                            color=colour,
                            alpha=0.15,
                            linewidth=0,
                        )
                for point, y in zip(series, ys, strict=True):
                    rows.append([case, topology, backend, point["bytes"], point["n"], y])

            ax.set_xscale("log", base=2)
            if value_axis == "latency":
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(topology)
            if col == 0:
                measure = f"latency ({sweep.unit})" if value_axis == "latency" else "GB/s (bus)"
                ax.set_ylabel(axis_label(case, measure))
            if row == len(cases) - 1:
                ax.set_xlabel("message size (bytes)")

    ordered = [backend for backend in sweep.backends() if backend in handles]
    figure_legend(fig, [handles[backend] for backend in ordered], ordered, theme)
    fig.suptitle(
        "Latency vs message size" if value_axis == "latency" else "Bandwidth vs message size",
        color=theme["text"],
        fontsize=12,
    )

    out = outdir / f"{stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    write_table(
        outdir / f"{stem}.csv",
        [
            "case",
            "topology",
            "backend",
            "bytes",
            "n",
            sweep.unit if value_axis == "latency" else "gbytes_per_s",
        ],
        rows,
    )
    return out


FIT_MEASURES = (
    ("alpha", "latency floor α", "{unit}"),
    ("binf_gbs", "peak bandwidth B∞", "GB/s"),
    ("nhalf_bytes", "n½ = α·B∞", "bytes"),
)


def draw_fit(
    fits: list[dict],
    theme: dict,
    benchmark: str | None,
    outdir: Path,
    stem: str,
    ext: str,
    include_single_rank: bool = False,
) -> Path | None:
    """The three alpha-beta numbers as grouped bars, one panel per measure.

    Three measures on one pair of axes would be a dual-axis chart, which invents
    a relationship between scales that share nothing; each gets its own panel.
    """
    fits = [fit for fit in fits if not benchmark or fit["benchmark"] == benchmark]
    fits = [fit for fit in fits if fit.get("alpha") is not None]
    # This figure reads fit.json rather than the points, so it needs the same
    # single-rank exclusion applied independently.
    if not include_single_rank:
        fits = [fit for fit in fits if not is_single_rank(fit["topology"])]
    if not fits:
        return None

    cases = sorted({fit["case"] for fit in fits})
    topologies = sorted({fit["topology"] for fit in fits}, key=topology_key)
    backends = [b for b in BACKEND_ORDER if b in {fit["backend"] for fit in fits}]
    index = {(fit["case"], fit["topology"], fit["backend"]): fit for fit in fits}
    unit = fits[0].get("unit", "us")

    fig, axes = panel_grid(len(cases), len(FIT_MEASURES), width=4.0, height=2.9)
    handles: dict[str, object] = {}
    rows: list[list] = []
    width = 0.8 / max(len(backends), 1)

    for row, case in enumerate(cases):
        for col, (field, label, unit_template) in enumerate(FIT_MEASURES):
            ax = axes[row][col]
            style_axes(ax, theme)
            for slot, backend in enumerate(backends):
                xs, ys = [], []
                for position, topology in enumerate(topologies):
                    fit = index.get((case, topology, backend))
                    value = fit.get(field) if fit else None
                    if value is None:
                        continue
                    xs.append(position - 0.4 + slot * width + width / 2)
                    ys.append(value)
                    if col == 0:
                        rows.append(
                            [
                                case,
                                topology,
                                backend,
                                fit.get("alpha"),
                                fit.get("binf_gbs"),
                                fit.get("nhalf_bytes"),
                                fit.get("tail_gbs"),
                                fit.get("points"),
                            ]
                        )
                if not xs:
                    continue
                # width * 0.88 leaves a surface gap between adjacent bars, which
                # is how they are separated - never a drawn border.
                bars = ax.bar(
                    xs,
                    ys,
                    width=width * 0.88,
                    color=colour_for(backend, theme),
                    label=backend,
                    linewidth=0,
                )
                handles.setdefault(backend, bars)
            ax.set_xticks(range(len(topologies)))
            ax.set_xticklabels(topologies)
            # n-half spans two orders of magnitude and needs a log axis; the
            # other two read better linear, without log minor-tick clutter.
            if field == "nhalf_bytes":
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(label)
            axis_unit = unit_template.format(unit=unit)
            ax.set_ylabel(axis_label(case, axis_unit) if col == 0 else axis_unit)

    ordered = [backend for backend in backends if backend in handles]
    figure_legend(fig, [handles[backend] for backend in ordered], ordered, theme)
    fig.suptitle("Latency floor, peak bandwidth, and the knee", color=theme["text"], fontsize=12)

    out = outdir / f"{stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    write_table(
        outdir / f"{stem}.csv",
        [
            "case",
            "topology",
            "backend",
            f"alpha_{unit}",
            "binf_gbytes_per_s",
            "nhalf_bytes",
            "tail_gbytes_per_s",
            "points",
        ],
        rows,
    )
    return out


def draw_heatmap(sweep: Sweep, theme: dict, outdir: Path, stem: str, ext: str) -> Path | None:
    """Speedup against the cuda_mpi baseline, backend x message size.

    Diverging scale: the midpoint is parity with the baseline and reads as
    neutral grey, the poles as opposite. The ramp is reversed so the faster pole
    is the cool one - the fastest backend must not be painted in the alarm
    colour. Every cell is also in the CSV, because a continuous colour scale
    must never be the only way to read a value.
    """
    lower_is_better = sweep.metric != "gbytes_per_s"
    panels = [
        (case, topo)
        for case in sweep.cases
        for topo in sweep.topologies
        if (case, topo) in sweep.curves
    ]
    if not panels:
        return None

    cmap = LinearSegmentedColormap.from_list("speedup", list(reversed(theme["diverging"])))
    cols = min(len(panels), len(sweep.topologies)) or 1
    rows_count = math.ceil(len(panels) / cols)
    fig, axes = panel_grid(rows_count, cols, width=3.8, height=3.0)

    rows: list[list] = []
    mesh = None
    ratios: list[float] = []
    grids: dict[tuple[str, str], tuple[list[str], list[int], list[list[float | None]]]] = {}

    for case, topology in panels:
        backends_here = sweep.curves[(case, topology)]
        base = {p["bytes"]: p["value_mean"] for p in backends_here.get(BASELINE_BACKEND, [])}
        sizes = sorted({p["bytes"] for series in backends_here.values() for p in series})
        listed = [b for b in sweep.backends() if b in backends_here]
        grid: list[list[float | None]] = []
        for backend in listed:
            by_size = {p["bytes"]: p["value_mean"] for p in backends_here[backend]}
            line: list[float | None] = []
            for size in sizes:
                value, reference = by_size.get(size), base.get(size)
                if not value or not reference:
                    line.append(None)
                    continue
                speedup = reference / value if lower_is_better else value / reference
                line.append(speedup)
                ratios.append(speedup)
                rows.append([case, topology, backend, size, speedup])
            grid.append(line)
        grids[(case, topology)] = (listed, sizes, grid)

    if not ratios:
        plt.close(fig)
        return None
    span = max(max(ratios), 1 / min(ratios)) if min(ratios) > 0 else max(ratios)
    norm = TwoSlopeNorm(vmin=1 / span, vcenter=1.0, vmax=span)

    for position, (case, topology) in enumerate(panels):
        ax = axes[position // cols][position % cols]
        listed, sizes, grid = grids[(case, topology)]
        data = [[float("nan") if cell is None else cell for cell in line] for line in grid]
        mesh = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        # Repeat the category labels only on the edges of the panel grid.
        ax.set_xticks(range(len(sizes)))
        bottom_row = position // cols == rows_count - 1
        ax.set_xticklabels(
            [format_bytes(size) for size in sizes] if bottom_row else [],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_yticks(range(len(listed)))
        ax.set_yticklabels(listed if position % cols == 0 else [], fontsize=8)
        ax.set_title(panel_title(case, topology))
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for empty in range(len(panels), rows_count * cols):
        axes[empty // cols][empty % cols].set_visible(False)

    bar = fig.colorbar(mesh, ax=axes, shrink=0.7, pad=0.02)
    bar.set_label(f"speedup vs {BASELINE_BACKEND} (>1 faster)", color=theme["muted"])
    bar.ax.tick_params(color=theme["muted"], labelcolor=theme["muted"])
    bar.outline.set_visible(False)
    fig.suptitle(
        f"Speedup against the {BASELINE_BACKEND} baseline", color=theme["text"], fontsize=12
    )

    out = outdir / f"{stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    write_table(
        outdir / f"{stem}.csv",
        ["case", "topology", "backend", "bytes", "speedup_vs_baseline"],
        rows,
    )
    return out


def select_size(sweep: Sweep, requested: str) -> int | None:
    """Which message size the distribution figure should show.

    The distribution is per message size, so one has to be picked. "min" is the
    default because the small-message end is the latency regime, which is where
    run-to-run spread actually matters; an explicit byte count snaps to the
    nearest swept size rather than silently plotting nothing.
    """
    sizes = sorted(
        {
            point["bytes"]
            for backends in sweep.curves.values()
            for series in backends.values()
            for point in series
        }
    )
    if not sizes:
        return None
    if requested == "min":
        return sizes[0]
    if requested == "max":
        return sizes[-1]
    return min(sizes, key=lambda size: abs(size - int(requested)))


def _run_box(run: dict, label: str) -> dict | None:
    """One job's five-number summary as a matplotlib bxp entry.

    Percentile fields are optional in the schema (results predating them exist),
    so a run missing the quartiles is dropped rather than guessed at.
    """
    if run.get("median") is None or run.get("p25") is None or run.get("p75") is None:
        return None
    return {
        "med": run["median"],
        "q1": run["p25"],
        "q3": run["p75"],
        # Whiskers are the extremes of the same reduced series the box comes
        # from, not a 1.5-IQR rule: these are the real min and max observed.
        "whislo": run.get("minimum", run["p25"]),
        "whishi": run.get("maximum", run["p75"]),
        "fliers": [],
        "label": label,
    }


def draw_distribution(
    sweep: Sweep, theme: dict, outdir: Path, stem: str, ext: str, requested_size: str = "min"
) -> tuple[Path, int] | None:
    """Per-job timing distributions at one message size, grouped by backend.

    Two dispersions are visible at once, and they mean different things: the
    height of a box is the spread of samples *within* one job, while the offset
    between a backend's boxes is the spread *across* independent allocations.
    Trials inside one job share an allocation, so only the latter says anything
    about reproducibility.
    """
    size = select_size(sweep, requested_size)
    if size is None:
        return None

    panels = [
        (case, topo)
        for case in sweep.cases
        for topo in sweep.topologies
        if (case, topo) in sweep.curves
    ]
    if not panels:
        return None

    cols = min(len(panels), len(sweep.topologies)) or 1
    rows_count = math.ceil(len(panels) / cols)
    fig, axes = panel_grid(rows_count, cols, width=3.8, height=3.0, sharey=True)

    rows: list[list] = []
    drew_anything = False

    for position, (case, topology) in enumerate(panels):
        ax = axes[position // cols][position % cols]
        style_axes(ax, theme)
        backends_here = sweep.curves[(case, topology)]
        listed = [b for b in sweep.backends() if b in backends_here]

        centres, labels = [], []
        for slot, backend in enumerate(listed):
            point = next((p for p in backends_here[backend] if p["bytes"] == size), None)
            centres.append(slot)
            labels.append(backend)
            if point is None:
                continue
            boxes = [
                box
                for box in (_run_box(run, run.get("job", "")) for run in point.get("runs", []))
                if box is not None
            ]
            if not boxes:
                continue

            colour = colour_for(backend, theme)
            width = 0.8 / len(boxes)
            positions = [slot - 0.4 + index * width + width / 2 for index in range(len(boxes))]
            artists = ax.bxp(
                boxes,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            for patch in artists["boxes"]:
                patch.set_facecolor(colour)
                patch.set_alpha(0.45)
                patch.set_edgecolor(colour)
                patch.set_linewidth(1.2)
            for element in ("whiskers", "caps"):
                for artist in artists[element]:
                    artist.set_color(colour)
                    artist.set_linewidth(1.2)
            for artist in artists["medians"]:
                artist.set_color(theme["text"])
                artist.set_linewidth(1.4)
            drew_anything = True

            for box in boxes:
                rows.append(
                    [
                        case,
                        topology,
                        backend,
                        size,
                        box["label"],
                        box["whislo"],
                        box["q1"],
                        box["med"],
                        box["q3"],
                        box["whishi"],
                    ]
                )

        # Every panel lists the same backends, so label the bottom row only.
        ax.set_xticks(centres)
        bottom_row = position // cols == rows_count - 1
        ax.set_xticklabels(labels if bottom_row else [], rotation=45, ha="right", fontsize=7)
        ax.set_title(panel_title(case, topology))
        if position % cols == 0:
            ax.set_ylabel(f"time per iteration ({sweep.unit})")

    if not drew_anything:
        plt.close(fig)
        return None

    for empty in range(len(panels), rows_count * cols):
        axes[empty // cols][empty % cols].set_visible(False)

    fig.suptitle(
        f"Timing distribution at {format_bytes(size)} — one box per job; "
        "box = within-run IQR, whiskers = observed min/max",
        color=theme["text"],
        fontsize=12,
    )

    out = outdir / f"{stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    write_table(
        outdir / f"{stem}.csv",
        ["case", "topology", "backend", "bytes", "job", "min", "p25", "median", "p75", "max"],
        rows,
    )
    return out, size


# Cases are a small categorical set (moe's three routings, halo_1d's two batch
# modes), so they take the leading palette slots -- the only three validated for
# all-pairs separation rather than just adjacent pairs.
CASE_SLOTS = (0, 1, 2, 3)


def draw_phases(
    sweep: Sweep, theme: dict, outdir: Path, stem: str, ext: str, requested_size: str = "max"
) -> tuple[Path, int] | None:
    """Where one step's time goes, stacked by phase, one panel per topology.

    Only cg_step carries a breakdown, and only when measured with
    GPU_BENCH_CG_PHASES=1, so this returns None rather than an empty figure when
    no point has one.

    The bars total the phase sum, which is the step's cost with no overlap. The
    reported per-iteration time is marked separately: the gap between the two is
    how much the unsplit loop overlaps its phases, and it is the reason the
    breakdown is measured in its own pass rather than in the reported loop.
    """
    size = select_size(sweep, requested_size)
    if size is None:
        return None

    panels = []
    for (case, topology), backends in sweep.curves.items():
        entries = []
        for backend in (b for b in sweep.backends() if b in backends):
            point = next(
                (p for p in backends[backend] if p["bytes"] == size and p.get("phases")), None
            )
            if point is not None:
                entries.append((backend, point))
        if entries:
            panels.append((case, topology, entries))
    if not panels:
        return None
    panels.sort(key=lambda item: (item[0], topology_key(item[1])))

    # Wider than the other figures' panels: the bars carry backend names on the
    # y axis and the legend has five entries, which clip at the default width.
    fig, axes = panel_grid(
        1,
        len(panels),
        width=5.2,
        height=0.55 * max(len(entry[2]) for entry in panels) + 1.8,
    )
    handles: dict[str, object] = {}
    for column, (case, topology, entries) in enumerate(panels):
        ax = axes[0][column]
        labels = [backend for backend, _ in entries]
        positions = list(range(len(entries)))
        left = [0.0] * len(entries)
        for phase in PHASE_ORDER:
            widths = [point["phases"].get(phase) or 0.0 for _, point in entries]
            bars = ax.barh(
                positions,
                widths,
                left=left,
                height=0.62,
                color=phase_colour(phase, theme),
                # A hairline in the surface colour separates adjacent segments
                # so a thin phase never merges into its neighbour.
                edgecolor=theme["surface"],
                linewidth=1.0,
            )
            handles.setdefault(phase, bars[0])
            left = [value + width for value, width in zip(left, widths, strict=True)]

        reported = [point.get("value_mean") for _, point in entries]
        marks = [
            (pos, value)
            for pos, value in zip(positions, reported, strict=True)
            if value is not None
        ]
        if marks:
            mark = ax.scatter(
                [value for _, value in marks],
                [pos for pos, _ in marks],
                marker="|",
                s=260,
                linewidths=2.0,
                color=theme["text"],
                zorder=3,
            )
            handles.setdefault("reported", mark)

        ax.set_yticks(positions, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"time per iteration ({sweep.unit})")
        ax.set_title(panel_title(case, topology))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        style_axes(ax, theme)

    # The bars total the no-overlap cost; the tick marks where the unsplit
    # loop actually landed, so the gap between them is the overlap.
    fig.suptitle(f"cg_step phase breakdown at {format_bytes(size)}")
    figure_legend(fig, list(handles.values()), list(handles.keys()), theme)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.{ext}"
    fig.savefig(path)
    plt.close(fig)

    write_table(
        outdir / f"{stem}.csv",
        ["case", "topology", "backend", "bytes", *PHASE_ORDER, "sum", "reported"],
        [
            [
                case,
                topology,
                backend,
                size,
                *[point["phases"].get(phase) for phase in PHASE_ORDER],
                point["phases"].get("total"),
                point.get("value_mean"),
            ]
            for case, topology, entries in panels
            for backend, point in entries
        ],
    )
    return path, size


def short_case(case: str) -> str:
    """`uniform,hidden=256` -> `uniform`. The qualifier is constant per run."""
    return case.split(",")[0] or "(default)"


def draw_cases(
    sweep: Sweep, theme: dict, outdir: Path, stem: str, ext: str, reference: float | None = None
) -> Path | None:
    """Metric per case, grouped by backend, one panel per topology.

    For a benchmark whose `case` axis is the point of the experiment -- moe's
    routing distributions, halo_1d's isolated/steady -- the size sweep is not
    where the story is. This puts the cases side by side per backend so the
    penalty one case pays relative to another is readable directly.

    `reference` draws a horizontal line on the ratio row: for moe it is the
    measured expert imbalance, so a bar reaching it means the backend is purely
    imbalance-limited and one below it means the backend absorbed some skew.
    """
    cases = sweep.cases
    if len(cases) < 2:
        return None
    topologies = sweep.topologies
    backends = sweep.backends()
    # The reference case leads, so the series read in order of departure from
    # it: uniform -> locality -> hotspot for moe, isolated -> steady for halo_1d.
    base = next((c for c in cases if "uniform" in c), cases[0])
    cases = [base] + [c for c in cases if c != base]

    fig, axes = panel_grid(2, len(topologies), width=3.4, height=3.0, sharey="row")
    handles: dict[str, object] = {}
    rows: list[list] = []
    width = 0.8 / max(len(cases), 1)

    for col, topology in enumerate(topologies):
        top, bottom = axes[0][col], axes[1][col]
        for ax in (top, bottom):
            style_axes(ax, theme)

        for slot, case in enumerate(cases):
            xs, ys, rs = [], [], []
            for position, backend in enumerate(backends):
                series = sweep.curves.get((case, topology), {}).get(backend)
                ref = sweep.curves.get((base, topology), {}).get(backend)
                if not series or not ref:
                    continue
                value, ref_value = series[0]["value_mean"], ref[0]["value_mean"]
                xs.append(position - 0.4 + slot * width + width / 2)
                ys.append(value)
                rs.append(value / ref_value if ref_value else 0.0)
                rows.append(
                    [
                        topology,
                        backend,
                        short_case(case),
                        value,
                        value / ref_value if ref_value else 0.0,
                    ]
                )
            if not xs:
                continue
            colour = theme["series"][CASE_SLOTS[slot % len(CASE_SLOTS)]]
            bars = top.bar(xs, ys, width=width * 0.88, color=colour, linewidth=0)
            bottom.bar(xs, rs, width=width * 0.88, color=colour, linewidth=0)
            handles.setdefault(short_case(case), bars)

        if reference is not None:
            bottom.axhline(reference, color=theme["text"], linewidth=1.0, linestyle=(0, (4, 3)))
            if col == 0:
                bottom.text(
                    -0.4,
                    reference,
                    f" load imbalance {reference:.2f}x",
                    va="bottom",
                    ha="left",
                    fontsize=7,
                    color=theme["text"],
                )
        bottom.axhline(1.0, color=theme["grid"], linewidth=1.0)

        for ax in (top, bottom):
            ax.set_xticks(range(len(backends)))
        top.set_xticklabels([])
        bottom.set_xticklabels(backends, rotation=45, ha="right", fontsize=7)
        # Absolute cost spans two orders of magnitude between intra- and
        # inter-node; a shared linear axis would flatten every intra-node panel.
        top.set_yscale("log")
        top.set_title(topology)
        if col == 0:
            top.set_ylabel(f"time per iteration ({sweep.unit})")
            bottom.set_ylabel(f"relative to {short_case(base)}")

    ordered = [short_case(c) for c in cases if short_case(c) in handles]
    figure_legend(fig, [handles[c] for c in ordered], ordered, theme)
    fig.suptitle(
        "Cost by case, and the penalty relative to the reference case",
        color=theme["text"],
        fontsize=12,
    )

    out = outdir / f"{stem}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    write_table(
        outdir / f"{stem}.csv",
        ["topology", "backend", "case", sweep.unit, f"relative_to_{short_case(base)}"],
        rows,
    )
    return out
