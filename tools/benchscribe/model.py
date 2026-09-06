from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    CUDA_MPI_CUDA_AWARE_ITER = "cuda_mpi_cuda_aware_iter"
    CUDA_NVSHMEM_DEVICE = "cuda_nvshmem_device"
    CUDA_NVSHMEM_ITER = "cuda_nvshmem_iter"
    CUDA_NVSHMEM = "cuda_nvshmem"
    SYCL_ONECCL_OSHMPI = "sycl_oneccl_oshmpi"
    SYCL_ONECCL = "sycl_oneccl"
    CUDA_NCCL = "cuda_nccl"
    CUDA_MPI = "cuda_mpi"
    SYCL_MPI = "sycl_mpi"
    OSHMPI = "oshmpi"


class OutputFormat(str, Enum):
    MARKDOWN = "md"
    CSV = "csv"
    JSON = "json"


class MetricName(str, Enum):
    USEC = "usec"
    TIME_PER_ITER_S = "time_per_iter_s"
    TIME_S = "time_s"
    GBYTES_PER_S = "gbytes_per_s"


class Status(str, Enum):
    OK = "OK"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    ERROR = "ERROR"


BASELINE_BACKEND = Backend.CUDA_MPI.value
KNOWN_BACKEND_NAMES = tuple(sorted((backend.value for backend in Backend), key=len, reverse=True))


@dataclass(frozen=True)
class MetricSpec:
    name: MetricName
    unit: str
    lower_is_better: bool


METRIC_SPECS = {
    MetricName.USEC: MetricSpec(MetricName.USEC, "us", True),
    MetricName.TIME_PER_ITER_S: MetricSpec(MetricName.TIME_PER_ITER_S, "s", True),
    MetricName.TIME_S: MetricSpec(MetricName.TIME_S, "s", True),
    MetricName.GBYTES_PER_S: MetricSpec(MetricName.GBYTES_PER_S, "GB/s", False),
}


@dataclass(frozen=True)
class ParsedReportLine:
    name: str
    fields: dict[str, str]


@dataclass(frozen=True)
class Measurement:
    backend: str
    benchmark: str
    topology: str
    n: int
    fields: dict[str, str]
    valid: bool
    case: str = ""
    status: Status = Status.OK
    # Which job produced this, and which trial within it. Trials inside one job
    # share an allocation, so only the job distinguishes independent samples.
    job: str = ""
    trial: str = ""


@dataclass(frozen=True, order=True)
class GroupKey:
    benchmark: str
    topology: str
    n: int
    case: str = ""


@dataclass(frozen=True)
class RunStats:
    """One job's measurement of one point.

    The percentile fields come from the within-run iteration distribution and
    are absent in results produced before the harness emitted them, so every
    one of them is optional.
    """

    job: str = ""
    mean: float | None = None
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    stddev: float | None = None
    iterations: int | None = None


@dataclass(frozen=True)
class AcrossRuns:
    """Spread of one point across independent jobs.

    This is the dispersion worth plotting: trials inside a single job share an
    allocation, so their agreement says little about reproducibility.
    """

    n_runs: int = 0
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    stddev: float | None = None


# The phases of one cg_step iteration, in execution order. Kept as a tuple so
# the reported order is the order the work happens in, not alphabetical.
CG_PHASE_NAMES = ("pack", "halo", "compute", "reduce")


@dataclass(frozen=True)
class PhaseBreakdown:
    """Mean microseconds in each phase of one step, averaged over trials.

    From `cg_step`'s opt-in second pass (`GPU_BENCH_CG_PHASES=1`), which
    synchronizes between phases. `total` is the benchmark's own
    `phase_sum_usec`: what the step would cost with no overlap between phases,
    so it exceeds the reported per-iteration time by however much the unsplit
    loop overlaps them. Every field is optional, because records measured
    without the pass carry none of them.
    """

    pack: float | None = None
    halo: float | None = None
    compute: float | None = None
    reduce: float | None = None
    total: float | None = None

    def phase(self, name: str) -> float | None:
        return getattr(self, name)

    def is_empty(self) -> bool:
        return all(getattr(self, name) is None for name in CG_PHASE_NAMES)


@dataclass(frozen=True)
class BackendSummary:
    backend: str
    metric_value: float | None
    metric_min: float | None
    bandwidth: float | None
    nbytes: int | None
    trials: int
    valid_all: bool
    status: Status = Status.OK
    runs: tuple[RunStats, ...] = ()
    across_runs: AcrossRuns | None = None
    phases: PhaseBreakdown | None = None


@dataclass(frozen=True)
class SummaryRow:
    benchmark: str
    topology: str
    n: int
    metric: MetricSpec
    backend: str
    value: float | None
    value_min: float | None
    bandwidth: float | None
    nbytes: int | None
    delta_pct_vs_base: float | None
    speedup_vs_base: float | None
    trials: int
    valid_all: bool
    case: str = ""
    status: Status = Status.OK
    phases: PhaseBreakdown | None = None


GroupedMeasurements = dict[tuple[str, str, int, str, str], list[Measurement]]
SummaryGroups = dict[GroupKey, dict[str, BackendSummary]]
