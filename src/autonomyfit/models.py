from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .evidence import BenchmarkEvidence, EvidenceMatch

Task = str
Objective = Literal["latency", "throughput", "accuracy", "power", "memory", "balanced"]
Verdict = Literal[
    "VERIFIED_FIT",
    "FEASIBLE",
    "BENCHMARK_REQUIRED",
    "CONSTRAINT_FAIL",
    "NO_FIT",
]
RegistrySource = Literal["remote", "cache", "bundled-fallback", "custom"]
EvidenceConfidence = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]


@dataclass(frozen=True)
class RuntimeCapability:
    name: str
    available: bool
    version: str | None = None
    detail: str | None = None
    provider: str | None = None
    verified: bool = True


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
    accelerator_type: str = "cpu"
    memory_topology: str = "system"
    supported_precisions: tuple[str, ...] = field(default_factory=tuple)
    max_power_w: float | None = None
    compute_tops: float | None = None
    hardware_limits: tuple[str, ...] = field(default_factory=tuple)
    software_stack: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccuracyMetric:
    name: str
    value: float
    dataset: str | None = None
    higher_is_better: bool = True


@dataclass(frozen=True)
class RegistryProvenance:
    source: RegistrySource
    registry_version: int | None = None
    generated_at: str | None = None
    expires_at: str | None = None
    loaded_at: str | None = None
    signature_verified: bool = False
    stale: bool = False
    etag: str | None = None
    warning: str | None = None
    registry_url: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelProfile:
    id: str
    display_name: str
    family: str
    task: Task
    params_m: float | None
    source_id: str
    source_url: str
    runtimes: tuple[str, ...]
    accuracy: AccuracyMetric | None = None
    flops_b: float | None = None
    input_size: int | None = None
    published_memory_gb: float | None = None
    memory_scope: str | None = None
    notes: str | None = None
    variant: str | None = None
    input_modalities: tuple[str, ...] = field(default_factory=tuple)
    output_modalities: tuple[str, ...] = field(default_factory=tuple)
    source_revision: str | None = None
    release_date: str | None = None
    last_checked: str | None = None
    supported_precisions: tuple[str, ...] = field(default_factory=tuple)
    quantizations: tuple[str, ...] = field(default_factory=tuple)
    license_spdx: str | None = None
    license_status: str = "unknown"
    license_source_url: str | None = None
    verification_status: str = "unknown"
    last_verified: str | None = None
    benchmark_refs: tuple[str, ...] = field(default_factory=tuple)
    experimental: bool = False


@dataclass(frozen=True)
class BenchmarkRecord:
    """Legacy schema-v1 benchmark record retained for compatibility only."""

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
    max_memory_gb: float | None = None
    max_params_m: float | None = None
    runtime: str | None = None
    precision: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    artifact_path: Path | None = None
    artifact_sha256: str | None = None
    family: str | None = None
    license_spdx: str | None = None
    license_status: str | None = None
    objective: Objective = "balanced"
    min_confidence: float | None = None
    verified_only: bool = False
    include_experimental: bool = False


@dataclass(frozen=True)
class ConfidenceBreakdown:
    score: float
    level: EvidenceConfidence
    hardware_match: float
    runtime_precision_match: float
    evidence_quality: float
    freshness: float
    revision_identity: float
    quantity_coverage: float
    unresolved_constraints: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Recommendation:
    model: ModelProfile
    verdict: Verdict
    score: float
    runtime: str
    precision: str
    estimated_memory_gb: float | None
    memory_evidence: str
    benchmark: BenchmarkEvidence | None
    evidence_match: EvidenceMatch | None
    evidence_confidence: EvidenceConfidence
    runtime_available: bool
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    registry: RegistryProvenance | None = None
    confidence: ConfidenceBreakdown | None = None
    unknowns: tuple[str, ...] = field(default_factory=tuple)
    measured: tuple[str, ...] = field(default_factory=tuple)
    estimated: tuple[str, ...] = field(default_factory=tuple)
    next_benchmark: str | None = None
    pareto_rank: int | None = None
    objective_rank: int | None = None
    dominates: tuple[str, ...] = field(default_factory=tuple)
    dominated_by: tuple[str, ...] = field(default_factory=tuple)
    objective: Objective = "balanced"

    @property
    def latency_ms(self) -> float | None:
        return self.benchmark.latency_ms if self.benchmark else None

    @property
    def fps(self) -> float | None:
        return self.benchmark.fps if self.benchmark else None
