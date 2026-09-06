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

## 🧰 Prerequisites

Only the **SYCL backends** need this. The CUDA presets build from cluster
modules alone -- skip to [Run](#-run) if that is all you need.

SYCL requires an
[intel/llvm source build with NVIDIA CUDA support](https://intel.github.io/llvm/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda);
stock oneAPI DPC++ has no NVPTX target. With hwloc installed, load the Leonardo
build modules and build DPC++ 6.3 once. The two paths below are what
`cluster/leonardo/env/sycl.sh` looks for by default, so using them leaves step
1️⃣ nothing to set:

```bash
module load gcc/12.2.0 cmake/4.1.2 ninja python cuda/12.2

export DPCPP_HOME="$HOME/opt/dpcpp_6.3"
export HWLOC_ROOT="$HOME/opt/hwloc"
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

The compiler must end up at `$DPCPP_HOME/llvm/build/install/bin/clang++`.

## ▶️ Run

Benchmark definitions are machine-independent; everything about a machine --
modules, compilers, CMake presets, communication-library tuning and scheduler
policy -- lives in its own directory under `cluster/`.

| Cluster | Scheduler | Accelerators | Status | Guide |
| --- | --- | --- | --- | --- |
| `leonardo` | Slurm | NVIDIA A100 | ✅ Validated | [`cluster/leonardo/README.md`](cluster/leonardo/README.md) |

`GPU_BENCH_CLUSTER` selects the machine and defaults to `leonardo`.

### 🚀 Leonardo in three steps

Run everything from the repository root on a login node.

**1️⃣ Set the environment.** Do this first -- `make init` reads these while it
builds. Skip the SYCL pair if you built DPC++ at the default paths above, and
skip the account if your Leonardo user has a site default.

```bash
export DPCPP_HOME="$HOME/opt/dpcpp_6.3"      # SYCL backends only
export HWLOC_ROOT="$HOME/opt/hwloc"          # SYCL backends only
export GPU_BENCH_SLURM_ACCOUNT=<account>
```

| Variable | Default | Set it when |
| --- | --- | --- |
| `DPCPP_HOME` | `$HOME/opt/dpcpp_6.3` | DPC++ was built elsewhere |
| `HWLOC_ROOT` | `$HOME/opt/hwloc` | hwloc was installed elsewhere |
| `DPCPP_INSTALL` | `$DPCPP_HOME/llvm/build/install` | the compiler is not under `llvm/build/install` |
| `GPU_BENCH_SLURM_ACCOUNT` | site default | `sbatch` rejects jobs with no account |
| `GPU_BENCH_SLURM_PARTITION` | `boost_usr_prod` | using another partition |
| `GPU_BENCH_WORK_ROOT` | `$SCRATCH/gpu-comm-bench` | sources and build trees go elsewhere |
| `GPU_BENCH_PREFIX_ROOT` | `$HOME/opt/gpu-comm-bench` | installed dependencies go elsewhere |

`env/sycl.sh` derives `CC`/`CXX` from `DPCPP_HOME` without checking that the
compiler is there, so a wrong value is not caught until CMake fails to find it
part-way through `make init`. Set it before you build, and re-export it in any
new shell.

**2️⃣ Bootstrap and build.** One command: it fetches and builds the dependencies,
then builds every preset.

```bash
make init
```

**3️⃣ Submit.** Check a cell resolves, run it, then run the matrix.

```bash
cluster/harness/launch.sh --explain allreduce cuda_mpi 1n4g   # resolve, no submit
cluster/harness/launch.sh allreduce cuda_mpi 1n4g             # one cell
cluster/harness/launch.sh --all allreduce                     # every backend and topology
```

`--explain` prints the binary, results path, and every environment variable with
the file that set it. Read it before a large submission; add `--dry-run --all`
to see what would go to the queue. The
[harness guide](cluster/harness/README.md) covers filters, overrides, result
paths, and profiling, and
[communication libraries](cluster/leonardo/README.md#-communication-libraries)
covers swapping a library version.

### ➕ Add another cluster

Create `cluster/<name>/cluster.sh` plus the `env/` and `runtime/` files it
dispatches to, and register the machine's backends in `backends.sh`. The harness
reaches a machine only through that one entry point, so adding a cluster must
not require any change under `cluster/harness/`. The required functions are
listed in [adding a cluster](cluster/harness/README.md#adding-a-cluster).

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
