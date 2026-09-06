# Clusters

The experiment system separates benchmark definitions from machine setup:

```text
cluster/
  harness/      shared launcher, matrix, jobs, and experiment defaults
  <name>/       modules, build presets, runtime tuning, and scheduler policy
```

Set `GPU_BENCH_CLUSTER=<name>` to select a machine. The shared launcher defaults
to `leonardo`.

| Cluster | Scheduler | Accelerators | Status |
| --- | --- | --- | --- |
| [`leonardo`](leonardo/README.md) | Slurm | NVIDIA A100 | Validated |

The [harness guide](harness/README.md) documents launcher syntax, result paths,
and the interface required to add another cluster.

## 🎛️ What you can change

Decide which of the three you are changing before you touch anything -- they
have very different blast radii.

| # | To change | Do this | Affects |
| --- | --- | --- | --- |
| 1️⃣ | 🔧 Runtime tuning: transport, service level, collective path | set the variable, or edit `<name>/runtime/*.sh` | one run |
| 2️⃣ | 📦 The version of a communication library | `export GPU_BENCH_<LIB>_VERSION=<version>` | one run, path-keyed |
| 3️⃣ | 🧰 The toolchain: compilers, CUDA, module set | edit `<name>/env/<stack>.sh` | every backend on that stack |

### 📦 A/B a library version

Use this whenever the question is "does this library version change the answer".
Set one variable and the build tree and results tree separate themselves, so the
two runs cannot share a binary or be averaged into one cell:

```bash
export GPU_BENCH_NVSHMEM_VERSION=3.7.2

cluster/harness/launch.sh --explain halo_1d cuda_nvshmem 2n4g   # confirm first
cluster/harness/launch.sh --all halo_1d
```

`--explain` prints the resolved library, build path and results path before
anything is submitted -- read it, then submit. Unset the variable to return to
the default; no cleanup is needed in between.

Each machine decides which of its libraries work this way, and the list is
deliberately small. Leonardo supports it for NVSHMEM only -- see its
[communication libraries](leonardo/README.md#-communication-libraries) table for
every other library it links and the recipe for changing that one.
