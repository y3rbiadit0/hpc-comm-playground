# All-To-All

`alltoall` measures a personalized exchange: every rank sends a distinct
GPU-resident block of float32 elements to every rank and validates the complete
permutation locally.

The complete operation, timing, validation, and bandwidth contract is in
[`docs/benchmarks/alltoall.md`](../../../../docs/benchmarks/alltoall.md).

## Configuration

```text
<max_count_per_peer> [iterations] [warmup] [comma-separated counts]
```

The harness defaults to 65,536 maximum elements per peer, 100 timed iterations,
and 20 warmup iterations. Without `GPU_BENCH_MSG_SIZES`, counts sweep powers of
two. Buffers contain `ranks * count_per_peer` elements.

Declared topologies are `1n1g`, `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, and
`8n4g`.

## Run

```bash
cluster/harness/launch.sh alltoall cuda_mpi 1n4g
GPU_BENCH_MSG_SIZES=1,256,65536 \
  cluster/harness/launch.sh alltoall cuda_nccl 2n4g
```

## UCC

The harness enables Open MPI UCC by default. Measured on Leonardo across every
declared topology (`cuda_mpi` and `sycl_mpi` agree within 1-2%):

| Regime | Effect of UCC |
| --- | --- |
| `1n1g` | none, as expected at one rank |
| `1n2g`, `2n1g`, all sizes | 1.4x-3x faster |
| `1n4g`, up to 8 KB/peer | 1.1x-2x faster |
| `4n4g`/`8n4g`, up to 256 B/peer | 17x-34x faster |
| `2n4g`, above 16 KB/peer | 1.4x-1.8x slower |
| `4n4g`, above 16 KB/peer | 1.1x-1.6x slower |
| `8n4g`, above 64 KB/peer | 1.07x-1.14x slower |

Without UCC, Open MPI's tuned alltoall collapses at small messages once the job
reaches 16 ranks -- 338-422 us at `4n4g` and 670-974 us at `8n4g` for 256 B per
peer, against 14-31 us with UCC -- and recovers at 512 B per peer. The
large-message regression is real but bounded, so the default trades at most 1.8x
against as much as 34x.

Override per run when measuring the large-message end:

```bash
OMPI_MCA_coll_ucc_enable=0 cluster/harness/launch.sh alltoall cuda_mpi 2n4g
```
