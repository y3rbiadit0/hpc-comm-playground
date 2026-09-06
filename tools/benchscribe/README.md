# Benchscribe

Benchscribe scans benchmark stdout files, parses standardized `key=value`
records, aggregates repeated runs, and compares each backend with the
`cuda_mpi` baseline. It requires only Python 3.10 or newer and the standard
library.

## Use

```bash
python3 tools/benchscribe
python3 tools/benchscribe results --benchmark allreduce
python3 tools/benchscribe --format csv > summary.csv
python3 tools/benchscribe --metric gbytes_per_s --format csv > bandwidth.csv
python3 tools/benchscribe results --benchmark halo_1d --fit
```

Available output formats are Markdown, CSV, and JSON. The default results path is
`results/`; the default format is Markdown.

## Aggregation

Benchscribe:

- scans `**/*-stdout.txt` below the selected results directory;
- recovers topology from the result path;
- groups by benchmark, case, topology, and problem size;
- keeps MoE hidden width as part of the displayed case;
- averages valid trials and reports each backend relative to `cuda_mpi`;
- uses `cuda_mpi / backend` for latency speedup, where values above one are
  faster;
- excludes errors and validation failures from numeric summaries; and
- preserves `NOT_IMPLEMENTED`/`SKIP` capability records as `N/I`.

The process record contract is documented in
[`docs/reference/output-schema.md`](../../docs/reference/output-schema.md).

## Phase Breakdown

`cg_step` can emit a per-phase breakdown (`GPU_BENCH_CG_PHASES=1`). `--phases`
reports it, in any of the three formats:

```bash
python3 tools/benchscribe results --phases
python3 tools/benchscribe results --phases --format csv > phases.csv
```

Each row carries the four phases, their sum, the reported per-iteration time,
and the difference between the last two. That difference is the point of the
view: the breakdown comes from a second pass that synchronizes between phases,
so its sum is the step's cost with no overlap, while the reported time is what
the unsplit loop achieved. A large overlap means the backend pipelines the step
(NCCL, whose whole step is one asynchronous stream); a near-zero overlap means it
does not.

Points measured without the pass carry no breakdown at all, which is distinct
from a breakdown of zero, and they are omitted from this view rather than shown
as blank. `points.json` gains an optional `phases` object, null wherever the pass
did not run - an additive field, so `schema_version` stays at 1.

## Characterization

For message-size sweeps, `--fit` reduces each backend curve to descriptive
latency/bandwidth values without regression:

| Value | Definition |
| --- | --- |
| `alpha` | Median latency for messages up to 4 KiB, or the smallest available point |
| `B_inf` | Best reported bandwidth |
| `n_half` | `alpha * B_inf`, the estimated half-bandwidth message size |
| `peak_at` | Message size at peak bandwidth |
| `tail` | Bandwidth at the largest message size |

Bandwidth keeps each benchmark's convention. Ping-pong reports one-way
bandwidth; halo reports aggregate exchange bus bandwidth. Compare only like
conventions.

## JSON Contract

JSON retains per-job iteration distributions in `runs[]` and independent-job
spread in `across_runs`. Fit JSON carries the corresponding characterization
data. Both documents include `schema_version`, which is checked by
`gpu-bench-plot`; incompatible changes require coordinated version updates.

## Test

```bash
python3 -m unittest discover -s tools/benchscribe/tests
```
