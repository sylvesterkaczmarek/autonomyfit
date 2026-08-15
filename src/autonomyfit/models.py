from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Task = Literal["detection", "vlm"]
Verdict = Literal[
    "VERIFIED_FIT",
    "FEASIBLE",
    "BENCHMARK_REQUIRED",
    "CONSTRAINT_FAIL",
    "NO_FIT",
]


@dataclass(frozen=True)
class RuntimeCapability:
    name: str
    available: bool
    version: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class HardwareProfile:
    platform: str
    os_name: str
    architecture: str
    cpu: str
    ram_total_gb: float
    ram_available_gb: float
    gpu: str | None = None
    accelerator_memory_gb: float | None = None
    unified_memory: bool = False
    jetpack: str | None = None
    power_mode: str | None = None
    driver: str | None = None
    matched_profile: str | None = None
    runtimes: tuple[RuntimeCapability, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccuracyMetric:
    name: str
    value: float
    dataset: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    id: str
    display_name: str
    family: str
    task: Task
    params_m: float
    source_id: str
    source_url: str
    runtimes: tuple[str, ...]
    accuracy: AccuracyMetric | None = None
    flops_b: float | None = None
    input_size: int | None = None
    published_memory_gb: float | None = None
    memory_scope: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class BenchmarkRecord:
    hardware_id: str
    model_id: str
    runtime: str
    precision: str
    latency_ms: float
    source_id: str
    source_url: str
    measured: bool = True
    power_w: float | None = None

    @property
    def fps(self) -> float:
        return 1000.0 / self.latency_ms


@dataclass(frozen=True)
class Constraints:
    task: Task = "detection"
    min_fps: float | None = None
    max_latency_ms: float | None = None
    max_power_w: float | None = None
    min_accuracy: float | None = None
    runtime: str | None = None
    precision: str | None = None


@dataclass(frozen=True)
class Recommendation:
    model: ModelProfile
    verdict: Verdict
    score: float
    runtime: str
    precision: str
    estimated_memory_gb: float
    memory_evidence: str
    benchmark: BenchmarkRecord | None
    runtime_available: bool
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def latency_ms(self) -> float | None:
        return self.benchmark.latency_ms if self.benchmark else None

    @property
    def fps(self) -> float | None:
        return self.benchmark.fps if self.benchmark else None
