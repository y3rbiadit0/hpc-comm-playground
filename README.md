# GPU Communication Benchmark

GPU Communication Benchmark compares communication models with equivalent
GPU-resident workloads. The suite runs the same communication patterns through
MPI, collective libraries, and one-sided SHMEM implementations so their latency,
bandwidth, scaling, and application-level behavior can be compared directly.

The project currently targets NVIDIA GPUs and is validated on the Leonardo
supercomputer. CUDA and SYCL implementations are provided where the underlying
libraries support both programming models.

## 📊 Benchmarks

| Benchmark | Kind | Measures |
| --- | --- | --- |
| [`pingpong`](docs/benchmarks/pingpong.md) | Microbenchmark | Two-endpoint one-way latency and bandwidth |
| [`halo_1d`](docs/benchmarks/halo-1d.md) | Microbenchmark | Neighbor exchange in a periodic ring |
| [`allreduce`](docs/benchmarks/allreduce.md) | Microbenchmark | Collective sum latency and bandwidth |
| [`alltoall`](docs/benchmarks/alltoall.md) | Microbenchmark | Personalized exchange and bus bandwidth |
| [`cg_step`](docs/benchmarks/cg-step.md) | Application pattern | Stencil halo exchange followed by two reductions |
| [`moe`](docs/benchmarks/moe.md) | Application pattern | Variable, skewed mixture-of-experts traffic |

Microbenchmarks sweep message size. Application patterns use a fixed problem
size and sweep rank count, testing whether isolated communication results predict
the behavior of a mixed workload. The rationale for this suite is documented in
[`docs/design/benchmark-selection.md`](docs/design/benchmark-selection.md).
Defaults, supported topologies, and launch examples are in the
[experiment operations](cluster/harness/README.md#experiment-operations).

## 🔀 Backend implementations

| Model | Backend | Implementation |
| --- | --- | --- |
| MPI | CUDA-aware MPI | [`src/mpi/cuda`](src/mpi/cuda/README.md) |
| MPI | SYCL + CUDA-aware MPI | [`src/mpi/sycl`](src/mpi/sycl/README.md) |
| XCCL | NCCL | [`src/xccl/cuda`](src/xccl/cuda/README.md) |
| XCCL | oneCCL | [`src/xccl/sycl`](src/xccl/sycl/README.md) |
| SHMEM | NVSHMEM | [`src/shmem/nvshmem`](src/shmem/nvshmem/README.md) |
| SHMEM | OSHMPI | [`src/shmem/oshmpi`](src/shmem/oshmpi/README.md) |

See the [support matrix](docs/reference/support-matrix.md) for benchmark, backend, and
topology coverage.

## 🖥️ Clusters

Benchmark definitions are machine-independent; everything about a machine --
modules, compilers, CMake presets, communication-library tuning and scheduler
policy -- lives in its own directory under `cluster/`.

| Cluster | Scheduler | Accelerators | Status | Setup guide |
| --- | --- | --- | --- | --- |
| `leonardo` | Slurm | NVIDIA A100 | ✅ Validated | [`cluster/leonardo/README.md`](cluster/leonardo/README.md) |

Select a machine with `GPU_BENCH_CLUSTER=<name>`; the launcher defaults to
`leonardo`.

### 🚀 Set up Leonardo

From a login node, working in the repository root:

| Step | Where |
| --- | --- |
| 1️⃣ Install the two SYCL prerequisites (DPC++ with NVPTX, hwloc) | [Prerequisites](cluster/leonardo/README.md#-prerequisites) |
| 2️⃣ Bootstrap dependencies and build every preset | [Build](cluster/leonardo/README.md#-build) |
| 3️⃣ Set a Slurm account and submit | [Run](cluster/leonardo/README.md#-run) |

Only the SYCL backends need step 1 -- the CUDA presets build from cluster
modules alone.

### ➕ Add another cluster

Create `cluster/<name>/cluster.sh` plus the `env/` and `runtime/` files it
dispatches to, and register the machine's backends in `backends.sh`. The harness
reaches a machine only through that one entry point, so adding a cluster must
not require any change under `cluster/harness/`. The required functions are
listed in [adding a cluster](cluster/harness/README.md#adding-a-cluster).

## 🔧 Build

### Build DPC++

The SYCL backends require an
[intel/llvm source build with NVIDIA CUDA support](https://intel.github.io/llvm/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda).
With hwloc installed under `$HOME/local/hwloc`, load the Leonardo build modules
and build DPC++ 6.3 once:

```bash
module load gcc/12.2.0 cmake/4.1.2 ninja python cuda/12.2

export DPCPP_HOME="$HOME/opt/dpcpp"
export HWLOC_ROOT="$HOME/local/hwloc"
mkdir -p "$DPCPP_HOME"
git clone https://github.com/intel/llvm -b v6.3.0 --depth=1 \
  "$DPCPP_HOME/llvm"

CC=gcc CXX=g++ python "$DPCPP_HOME/llvm/buildbot/configure.py" \
  --cuda \
  -DCUDA_Toolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_PREFIX_PATH="$HWLOC_ROOT" \
  -DLIBHWLOC_INCLUDE_DIRS="$HWLOC_ROOT/include" \
  -DLIBHWLOC_LIBRARIES="$HWLOC_ROOT/lib/libhwloc.so"

CC=gcc CXX=g++ python "$DPCPP_HOME/llvm/buildbot/compile.py"
```

### Initialize The Project

With `DPCPP_HOME` and `HWLOC_ROOT` still exported, initialize the dependency
stack and build every benchmark preset with:

```bash
make init
```

For prerequisite paths, dependency targets, and validated toolchains, follow the
[Leonardo build guide](cluster/leonardo/README.md#-build).

## ▶️ Run

Submission commands belong to each [cluster guide](cluster/README.md). The shared
launcher accepts a benchmark, backend, and topology:

```bash
cluster/harness/launch.sh --explain allreduce cuda_mpi 1n4g
cluster/harness/launch.sh allreduce cuda_mpi 1n4g
```

`--explain` resolves a cell without submitting it. See the
[harness guide](cluster/harness/README.md) for matrix runs, overrides, result
paths, and adding another cluster.

## 📈 Analyze Results

Benchmark processes emit standardized `key=value` records. Benchscribe collects
records from `results/`, and the plot package consumes its JSON output:

```bash
python3 tools/benchscribe --benchmark allreduce
python3 tools/benchscribe --format json > points.json
uv run --project tools/plot gpu-bench-plot \
  --points points.json --benchmark allreduce --outdir figures
```

See [`tools/README.md`](tools/README.md) for the analysis workflow and
[`docs/reference/output-schema.md`](docs/reference/output-schema.md) for the report contract.

## 📚 Project Documentation

- [`docs/README.md`](docs/README.md): benchmark contracts, design, reference, and analysis
- [`cluster/README.md`](cluster/README.md): cluster integration and supported systems
- [`CHANGELOG.md`](CHANGELOG.md): release history and pending changes
- [`CONTRIBUTING.md`](CONTRIBUTING.md): development and contribution workflow
- [`LICENSE`](LICENSE): MIT license
