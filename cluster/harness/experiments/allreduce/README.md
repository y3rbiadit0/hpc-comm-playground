# Allreduce

`allreduce` measures an element-wise float32 sum over GPU-resident buffers. Every
rank contributes the same element count and receives the result.

The complete operation, timing, validation, and bandwidth contract is in
[`docs/benchmarks/allreduce.md`](../../../../docs/benchmarks/allreduce.md).

## Configuration

```text
<max_elements> [iterations] [warmup] [comma-separated message sizes]
```

The harness defaults to 4,194,304 maximum elements, 100 timed iterations, and 20
warmup iterations. Without `GPU_BENCH_MSG_SIZES`, each implementation sweeps
powers of two.

Declared topologies are `1n1g`, `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, and
`8n4g`. The single-rank case is a control rather than a network measurement.

## Run

```bash
cluster/harness/launch.sh allreduce cuda_mpi 1n4g
GPU_BENCH_MSG_SIZES=1,1024,4194304 \
  cluster/harness/launch.sh allreduce cuda_nccl 2n4g
```

On Leonardo, OSHMPI defaults to direct device-buffer reduction and requires UCC.
Set `GPU_BENCH_OSHMPI_ALLREDUCE_MEM=staged` to measure its explicit host-staging
fallback; the selected path is included in the result.

## UCC

The harness enables Open MPI UCC by default, but the effect is not uniform.
Measured across every declared topology (`cuda_mpi` and `sycl_mpi` agree within
1-2%):

| Regime | Effect of UCC |
| --- | --- |
| `1n2g`, `1n4g`, `2n1g`, all sizes | 1.1x-140x faster |
| `2n4g`/`4n4g`/`8n4g`, below 64 KB | 1.06x-2.2x **slower** |
| `2n4g`/`4n4g`/`8n4g`, at and above 64 KB | 1.05x-35x faster |

Two features of the curve are worth knowing before reading a result:

- **A cliff at 4 KB** on every topology with 4 GPUs per node. Latency roughly
  doubles crossing 2 KB to 4 KB with UCC on (+81% at `1n4g`, +120% at `8n4g`)
  while the non-UCC path is flat across the same step. Both backends reproduce
  it, so it is a UCC algorithm switch rather than noise.
- **A crossover at 64 KB** on multi-node topologies, below which UCC costs more
  than it saves. The default keeps UCC because the losses cap near 2.2x while
  the wins reach 140x, not because it wins everywhere.

Override per run to measure the small-message end:

```bash
OMPI_MCA_coll_ucc_enable=0 cluster/harness/launch.sh allreduce cuda_mpi 4n4g
```

The `1n1g` control does not currently measure what its name suggests: at one
rank Open MPI stages the reduction through host memory (16 MB in 21.4 ms, about
0.78 GB/s), which is 92x slower than the same buffer reduced across two GPUs.
Read it as a known artifact, not as the floor.
