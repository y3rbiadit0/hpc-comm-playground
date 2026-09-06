# Analysis Tools

| Tool | Purpose | Requirements |
| --- | --- | --- |
| [`benchscribe`](benchscribe/README.md) | Parse, aggregate, compare, and characterize benchmark results | Python 3.10+, standard library only |
| [`gpu-bench-plot`](plot/README.md) | Draw latency, bandwidth, fit, distribution, and speedup figures | Python 3.10+, `uv` |
| [`roofline`](roofline/README.md) | Plot kernel rooflines from Nsight Compute CSV exports | Python 3.10+, `uv` |
| [`ucc-ab.py`](ucc-ab.py) | Join two Benchscribe CSVs cell by cell to compare one environment change | Python 3.10+, standard library only |

## Results Workflow

Run Benchscribe from the repository root, then pass its versioned JSON to the
plot package:

```bash
python3 tools/benchscribe results --format json > points.json
python3 tools/benchscribe results --fit --format json > fit.json

uv run --project tools/plot gpu-bench-plot \
  --points points.json --fit fit.json \
  --benchmark halo_1d --figure all --outdir figures
```

Benchscribe is the only component that parses job output and computes the
latency/bandwidth characterization. Plotters consume its JSON rather than
reimplementing those rules.

## A/B Comparisons

Benchscribe compares backends *within* one results tree against the `cuda_mpi`
baseline, so an A/B on an environment variable needs two trees and a join.
`GPU_BENCH_RESULTS_ROOT` separates the trees at submission time -- a directory
suffix would not, because Benchscribe recovers the topology by searching the
whole path:

```bash
GPU_BENCH_RESULTS_ROOT=results-ucc-on  OMPI_MCA_coll_ucc_enable=1 \
  cluster/harness/launch.sh --all alltoall
GPU_BENCH_RESULTS_ROOT=results-ucc-off OMPI_MCA_coll_ucc_enable=0 \
  cluster/harness/launch.sh --all alltoall

python3 tools/benchscribe results-ucc-on  --format csv > ucc-on.csv
python3 tools/benchscribe results-ucc-off --format csv > ucc-off.csv
python3 tools/ucc-ab.py ucc-on.csv ucc-off.csv > UCC-AB.md
```

`ucc-ab.py` joins on (benchmark, case, topology, backend, n) and reports the
ratio, the percent change, and a per-cell verdict, warning on stderr about any
cell present in only one arm. `--format csv` emits the join as data and
`--threshold` sets what counts as no effect.

To exercise the pipeline without a cluster, generate explicitly synthetic data:

```bash
uv run --project tools/plot gpu-bench-plot-sample /tmp/demo-results
python3 tools/benchscribe /tmp/demo-results --format json > /tmp/points.json
uv run --project tools/plot gpu-bench-plot \
  --points /tmp/points.json --outdir /tmp/figures
```

Synthetic results are for tool testing only and must not be published as
measurements.
