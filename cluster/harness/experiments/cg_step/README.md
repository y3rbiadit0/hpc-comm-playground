# Conjugate-Gradient Step

`cg_step` extracts the communication shape of one conjugate-gradient iteration:
a five-point stencil over a column-slab decomposition, one neighbor halo
exchange, and two scalar global sums. It is an application pattern, not a full
solver.

The complete operation, timing, validation, and metric contract is in
[`docs/benchmarks/cg-step.md`](../../../../docs/benchmarks/cg-step.md).

## Configuration

```text
<grid_side> [iterations] [warmup]
```

The harness defaults to a `512 x 512` global grid, 50 timed iterations, and 10
warmup iterations. The small grid keeps communication visible relative to the
stencil kernel.

Declared topologies are `1n1g`, `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, and
`8n4g`. At one rank, the benchmark provides a compute and local-reduction
baseline.

## Run

```bash
cluster/harness/launch.sh cg_step cuda_mpi 1n4g
GPU_BENCH_N=1024 GPU_BENCH_NTRIALS=1 \
  cluster/harness/launch.sh cg_step cuda_nvshmem 2n4g

# Sweep the grid side. GPU_BENCH_MSG_SIZES carries grid sides here, not message
# sizes, and GPU_BENCH_N must be at least the largest of them.
GPU_BENCH_N=2048 GPU_BENCH_MSG_SIZES=256,512,1024,2048 \
  cluster/harness/launch.sh cg_step cuda_mpi 4n4g

# Break the step into pack / halo / compute / reduce.
GPU_BENCH_CG_PHASES=1 cluster/harness/launch.sh cg_step cuda_mpi 4n4g
```

A bare run measures the single grid `GPU_BENCH_N` names; the side sets each
rank's column count, so sweeping it by default would change what "the cg_step
result" means. One record is emitted per side.

`GPU_BENCH_CG_PHASES=1` adds `phase_pack_usec`, `phase_halo_usec`,
`phase_compute_usec`, `phase_reduce_usec` and `phase_sum_usec` to the record.
They come from a second, more synchronized pass, so the headline `usec` is
unchanged and stays comparable with earlier results; see
[`docs/benchmarks/cg-step.md`](../../../../docs/benchmarks/cg-step.md). The pass
roughly doubles run time and is off by default.

The reductions contain one double each. Leonardo disables UCC for this
small-message operation; OSHMPI consequently defaults to its staged scalar path.
The allreduce topology sweep supports that default on multi-node topologies,
where UCC costs 25-49% at these sizes, but contradicts it on `1n2g`, `1n4g` and
`2n1g`, where UCC is 30-50% faster. Set the variable explicitly when the
single-node cells matter.
Set both `OMPI_MCA_coll_ucc_enable=1` and
`GPU_BENCH_OSHMPI_CG_REDUCE_MEM=device` to measure its device path explicitly.
