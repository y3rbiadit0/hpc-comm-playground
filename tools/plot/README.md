# GPU Bench Plot

Figures for the gpu-comm-bench results, drawn from Benchscribe JSON.

This is a `uv` project rather than a loose script, so the plotting dependencies
are declared and locked in one place and never leak into Benchscribe — which
stays deliberately stdlib-only with no build step.

## Contract with Benchscribe

The plotter reads JSON and nothing else. It does **not** parse Slurm output and
does **not** compute the α–β fit: Benchscribe owns both. Each input file carries
a `schema_version`, matched against `SUPPORTED_POINTS_SCHEMA` /
`SUPPORTED_FIT_SCHEMA` in `data.py`. A mismatch is a hard error naming both
versions — never widen the check to paper over a schema change.

## Use

```bash
# from the repository root
python3 tools/benchscribe results --format json       > points.json
python3 tools/benchscribe results --fit --format json > fit.json

uv run --project tools/plot gpu-bench-plot \
    --points points.json --fit fit.json \
    --benchmark halo_1d --figure all \
    --outdir docs/analysis/data/microbenchmarks/tuned/halo_1d/figures
```

`--project tools/plot` keeps the working directory where it is, so the paths you
pass stay relative to the repository root. Inside a synchronized project
environment, `uv run --project tools/plot python -m gpu_bench_plot` is equivalent.

| Figure | What it shows |
| --- | --- |
| `latency` | latency vs message size, log-log, panel per case × topology (`sweep` still accepted) |
| `bandwidth` | bus GB/s vs message size, same panel grid |
| `fit` | α, B∞ and n½ as grouped bars, one panel per measure (needs `--fit`) |
| `heatmap` | speedup vs the `cuda_mpi` baseline, backend × message size |
| `dist` | per-job timing distributions at one message size, as box plots (`--size`) |
| `phases` | `cg_step`'s per-phase breakdown at one size, stacked per backend (`--size`) |

Other options: `--figure` picks one of the above or `all`, `--theme light|dark`,
`--format svg|png|pdf`, `--benchmark` filters to one benchmark, and `--size`
chooses which message size the `dist` figure shows (`min` by default, `max`, or
a byte count snapped to the nearest swept size).

### Reading the `phases` figure

Bars total the phase sum, which is the step's cost with no overlap between
phases. The tick marks the reported per-iteration time, so the gap between the
bar end and the tick is how much the unsplit loop overlaps its phases. Only
`cg_step` measured with `GPU_BENCH_CG_PHASES=1` carries a breakdown; for anything
else the figure is skipped rather than drawn empty.

The four phase colours are their own slots, not the backend ones: a hue means a
backend in every other figure, and letting it also mean a phase here would make
the palette ambiguous across the figure set.

### Reading the `dist` figure

It shows the two dispersions Benchscribe keeps separate, and they answer
different questions:

- **Box height** — spread of samples *within* one job. For `halo_1d` a sample is
  a completed batch, so this is batch-to-batch jitter inside one allocation.
- **Offset between a backend's boxes** — spread *across* independent jobs. Trials
  inside one job share an allocation and the same GPUs, so only this one says
  anything about reproducibility, and it is usually the larger of the two.

Whiskers are the observed min and max of that same series, not a 1.5×IQR rule —
these runs have too few samples for the convention to mean anything. Runs from
results predating the percentile fields are dropped rather than guessed at; if
no run has quartiles the figure is skipped with a warning.

## Trying it without a cluster

```bash
uv run --project tools/plot gpu-bench-plot-sample /tmp/demo-results
python3 tools/benchscribe /tmp/demo-results --format json > /tmp/points.json
uv run --project tools/plot gpu-bench-plot --points /tmp/points.json --outdir /tmp/figures
```

That writes a synthetic results tree from a hand-written α–β model. **Its output
is not measured data** — it exists to exercise the pipeline offline. Never cite,
publish, or commit it.

## Design rules these figures follow

Two of them are load-bearing and easy to undo by accident:

- **Colour follows the backend, not its position.** `theme.py` maps each backend
  to a fixed palette slot, so filtering a figure down never repaints the
  survivors. The slot *order* is also the colourblind-safety mechanism: it was
  validated for adjacent-pair CVD separation against both surfaces (worst
  adjacent ΔE 9.1 light / 8.4 dark). Reordering the tuple invalidates that.
- **Every figure writes a companion `.csv`.** Three light-mode series colours sit
  below 3:1 contrast on the chart surface, and the palette's relief rule requires
  a non-colour path to every value wherever they are used. The CSV is it — and it
  is also what makes the speedup heatmap legible without colour at all.

Smaller ones, each commented at its site: no dual-axis panels (the three fit
measures get three panels); latency shares one log y-axis across all panels while
bandwidth shares per column, so inter-node panels do not flatten; the speedup
ramp is reversed so the fastest backend is not painted in the alarm colour.

## Development

```bash
uv sync --project tools/plot
uv run --project tools/plot pytest tools/plot/tests
uv run --project tools/plot ruff check tools/plot
```
