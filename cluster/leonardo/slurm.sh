#!/usr/bin/env bash

# SLURM submission defaults for Leonardo.
#
# sbatch reads SBATCH_ACCOUNT and SBATCH_PARTITION from the environment, so the
# job scripts carry no -A/-p directives. Set the account explicitly when your
# Slurm site does not provide a user default:
#
#   GPU_BENCH_SLURM_ACCOUNT=<account> cluster/harness/launch.sh --all allreduce
#
# Precedence is the usual SLURM one - an explicit `sbatch --account=...` still
# wins over anything set here.
#
# Sourced by cluster/leonardo/cluster.sh and by environment.sh, so both `make
# submit` and a hand-run sbatch from a prepared shell pick it up.

if [[ -n ${GPU_BENCH_SLURM_ACCOUNT:-} ]]; then
  export SBATCH_ACCOUNT=$GPU_BENCH_SLURM_ACCOUNT
fi
export SBATCH_PARTITION=${GPU_BENCH_SLURM_PARTITION:-${SBATCH_PARTITION:-boost_usr_prod}}
