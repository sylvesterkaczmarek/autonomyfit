from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator, FormatChecker
from platformdirs import user_data_path

EvidenceQuality = Literal[
    "local-measured",
    "standardized",
    "vendor-published",
    "third-party-reproducible",
    "metadata-estimate",
]

QUALITY_RANK: dict[str, int] = {
    "local-measured": 5,
    "standardized": 4,
    "vendor-published": 3,
    "third-party-reproducible": 2,
    "metadata-estimate": 1,
}


class EvidenceError(RuntimeError):
    """Base evidence-store error."""


class EvidenceSchemaError(EvidenceError):
    """Evidence data failed schema or semantic validation."""


@dataclass(frozen=True)
class LatencyStats:
    min_ms: float | None = None
    mean_ms: float | None = None
    median_ms: float | None = None
    p50_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    max_ms: float | None = None
    stdev_ms: float | None = None

    @property
    def representative_ms(self) -> float | None:
        return self.median_ms or self.p50_ms or self.mean_ms


@dataclass(frozen=True)
class PowerStats:
    mean_w: float | None = None
    max_w: float | None = None
    energy_j: float | None = None
    scope: str | None = None


@dataclass(frozen=True)
class BenchmarkEvidence:
    id: str
    model_id: str
    model_revision: str | None
    artifact_id: str | None
    artifact_sha256: str | None
    artifact_format: str | None
    hardware_id: str
    hardware_name: str | None
    runtime: str
    runtime_version: str | None
    provider: str | None
    precision: str
    quantization: str | None
    batch_size: int | None
    input_shapes: dict[str, list[int]]
    power_mode: str | None
    clocks: dict[str, str]
    warmup: int | None
    iterations: int | None
    latency: LatencyStats
    throughput_fps: float | None
    power: PowerStats
    peak_memory_mb: float | None
    peak_memory_scope: str | None
    quality: EvidenceQuality
    source_id: str
    source_url: str
    source_date: str | None
    software_stack_id: str | None
    notes: str | None = None
    verified_identity: bool = False

    @property
    def fps(self) -> float | None:
        if self.throughput_fps is not None:
            return self.throughput_fps
        latency = self.latency.representative_ms
        return 1000.0 / latency if latency and latency > 0 else None

    @property
    def latency_ms(self) -> float | None:
        return self.latency.representative_ms

    @property
    def evidence_label(self) -> str:
        return self.quality.replace("-", " ")

    @property
    def eligible_for_verified_fit(self) -> bool:
        return (
            self.quality in {"local-measured", "standardized"}
            and self.verified_identity
            and bool(self.model_revision)
            and bool(self.artifact_sha256)
        )


@dataclass(frozen=True)
class EvidenceMatch:
    evidence: BenchmarkEvidence
    exact: bool
    identity_complete: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvidenceStore:
    document: dict[str, Any]
    benchmarks: tuple[BenchmarkEvidence, ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_resource(name: str) -> dict[str, Any]:
    resource = files("autonomyfit.data").joinpath(name)
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceSchemaError(f"{name} must contain a JSON object")
    return value


def _validator(schema_name: str) -> Draft202012Validator:
    schema = _load_json_resource(schema_name)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _validate(document: dict[str, Any], schema_name: str, label: str) -> None:
    errors = sorted(
        _validator(schema_name).iter_errors(document),
        key=lambda item: list(item.path),
    )
    if not errors:
        return
    first = errors[0]
    location = ".".join(str(part) for part in first.absolute_path) or "document"
    raise EvidenceSchemaError(f"{label} schema error at {location}: {first.message}")


def validate_evidence_document(document: dict[str, Any]) -> None:
    _validate(document, "evidence-v2.schema.json", "evidence")
    ids: list[str] = []
    for collection in (
        "model_revisions",
        "artifacts",
        "accuracy_evidence",
        "compatibility_evidence",
        "memory_evidence",
        "benchmark_evidence",
        "hardware_profiles",
        "runtime_profiles",
        "software_stacks",
    ):
        ids.extend(item["id"] for item in document.get(collection, []))
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise EvidenceSchemaError(
            "evidence entity ids must be globally unique: " + ", ".join(duplicates)
        )

    model_ids = {item["id"] for item in document["models"]}
    revision_ids = {item["id"] for item in document["model_revisions"]}
    artifact_ids = {item["id"] for item in document["artifacts"]}
    hardware_ids = {item["id"] for item in document["hardware_profiles"]}
    runtime_ids = {item["id"] for item in document["runtime_profiles"]}
    stack_ids = {item["id"] for item in document["software_stacks"]}
    for item in document["model_revisions"]:
        if item["model_id"] not in model_ids:
            raise EvidenceSchemaError(
                f"model revision {item['id']} references unknown model"
            )
    for item in document["artifacts"]:
        if item["model_revision_id"] not in revision_ids:
            raise EvidenceSchemaError(
                f"artifact {item['id']} references unknown model revision"
            )
    for item in document["accuracy_evidence"]:
        if item["model_revision_id"] not in revision_ids:
            raise EvidenceSchemaError(f"accuracy {item['id']} references unknown model revision")
    for item in document["compatibility_evidence"]:
        if item["model_revision_id"] not in revision_ids:
            raise EvidenceSchemaError(f"compatibility {item['id']} references unknown model revision")
        if item["hardware_profile_id"] not in hardware_ids:
            raise EvidenceSchemaError(f"compatibility {item['id']} references unknown hardware")
        if item["runtime_profile_id"] not in runtime_ids:
            raise EvidenceSchemaError(f"compatibility {item['id']} references unknown runtime")
    for item in document["memory_evidence"]:
        if item["model_revision_id"] not in revision_ids:
            raise EvidenceSchemaError(f"memory {item['id']} references unknown model revision")
        if item["hardware_profile_id"] not in hardware_ids:
            raise EvidenceSchemaError(f"memory {item['id']} references unknown hardware")
    for item in document["benchmark_evidence"]:
        if item["model_revision_id"] not in revision_ids:
            raise EvidenceSchemaError(
                f"benchmark {item['id']} references unknown model revision"
            )
        artifact_id = item.get("artifact_id")
        if artifact_id is not None and artifact_id not in artifact_ids:
            raise EvidenceSchemaError(
                f"benchmark {item['id']} references unknown artifact"
            )
        if item["hardware_profile_id"] not in hardware_ids:
            raise EvidenceSchemaError(
                f"benchmark {item['id']} references unknown hardware"
            )
        if item["runtime_profile_id"] not in runtime_ids:
            raise EvidenceSchemaError(
                f"benchmark {item['id']} references unknown runtime profile"
            )
        stack_id = item.get("software_stack_id")
        if stack_id is not None and stack_id not in stack_ids:
            raise EvidenceSchemaError(
                f"benchmark {item['id']} references unknown software stack"
            )


def validate_benchmark_report(document: dict[str, Any]) -> None:
    _validate(document, "benchmark-report-v2.schema.json", "benchmark report")
    created = datetime.fromisoformat(document["created_at"].replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise EvidenceSchemaError("benchmark report created_at must include a timezone")
    if document["quality"] != "local-measured":
        raise EvidenceSchemaError("local benchmark reports must use quality=local-measured")
    artifact = document["artifact"]
    if not artifact.get("sha256"):
        raise EvidenceSchemaError("local benchmark reports require artifact.sha256")


def _latency_from_dict(value: dict[str, Any]) -> LatencyStats:
    return LatencyStats(
        min_ms=value.get("min_ms"),
        mean_ms=value.get("mean_ms"),
        median_ms=value.get("median_ms"),
        p50_ms=value.get("p50_ms"),
        p90_ms=value.get("p90_ms"),
        p95_ms=value.get("p95_ms"),
        p99_ms=value.get("p99_ms"),
        max_ms=value.get("max_ms"),
        stdev_ms=value.get("stdev_ms"),
    )


def _power_from_dict(value: dict[str, Any] | None) -> PowerStats:
    value = value or {}
    return PowerStats(
        mean_w=value.get("mean_w"),
        max_w=value.get("max_w"),
        energy_j=value.get("energy_j"),
        scope=value.get("scope"),
    )


def _bundled_benchmarks(document: dict[str, Any]) -> tuple[BenchmarkEvidence, ...]:
    revisions = {item["id"]: item for item in document["model_revisions"]}
    artifacts = {item["id"]: item for item in document["artifacts"]}
    hardware = {item["id"]: item for item in document["hardware_profiles"]}
    runtimes = {item["id"]: item for item in document["runtime_profiles"]}
    result: list[BenchmarkEvidence] = []
    for item in document["benchmark_evidence"]:
        revision = revisions[item["model_revision_id"]]
        artifact = artifacts.get(item.get("artifact_id"))
        hardware_profile = hardware[item["hardware_profile_id"]]
        runtime = runtimes[item["runtime_profile_id"]]
        result.append(
            BenchmarkEvidence(
                id=item["id"],
                model_id=revision["model_id"],
                model_revision=revision.get("revision"),
                artifact_id=artifact["id"] if artifact else None,
                artifact_sha256=artifact.get("sha256") if artifact else None,
                artifact_format=artifact.get("format") if artifact else None,
                hardware_id=hardware_profile["id"],
                hardware_name=hardware_profile.get("display_name"),
                runtime=runtime["runtime"],
                runtime_version=runtime.get("version"),
                provider=runtime.get("provider"),
                precision=runtime["precision"],
                quantization=runtime.get("quantization"),
                batch_size=item.get("batch_size"),
                input_shapes=item.get("input_shapes", {}),
                power_mode=item.get("power_mode"),
                clocks=item.get("clocks", {}),
                warmup=item.get("warmup"),
                iterations=item.get("iterations"),
                latency=_latency_from_dict(item["latency"]),
                throughput_fps=item.get("throughput_fps"),
                power=_power_from_dict(item.get("power")),
                peak_memory_mb=item.get("peak_memory_mb"),
                peak_memory_scope=item.get("peak_memory_scope"),
                quality=item["quality"],
                source_id=item["source_id"],
                source_url=item["source_url"],
                source_date=item.get("source_date"),
                software_stack_id=item.get("software_stack_id"),
                notes=item.get("notes"),
                verified_identity=bool(item.get("verified_identity", False)),
            )
        )
    return tuple(result)


def benchmark_evidence_from_report(document: dict[str, Any]) -> BenchmarkEvidence:
    validate_benchmark_report(document)
    model = document["model"]
    artifact = document["artifact"]
    hardware = document["hardware"]
    software = document["software"]
    execution = document["execution"]
    metrics = document["metrics"]
    return BenchmarkEvidence(
        id=document["benchmark_id"],
        model_id=model["id"],
        model_revision=model.get("revision"),
        artifact_id=f"local:{artifact['sha256'][:20]}",
        artifact_sha256=artifact["sha256"],
        artifact_format=artifact.get("format"),
        hardware_id=hardware["id"],
        hardware_name=hardware.get("device"),
        runtime=software["runtime"],
        runtime_version=software.get("runtime_version"),
        provider=software.get("provider"),
        precision=execution["precision"],
        quantization=execution.get("quantization"),
        batch_size=execution.get("batch_size"),
        input_shapes=execution.get("input_shapes", {}),
        power_mode=hardware.get("power_mode"),
        clocks=hardware.get("clocks", {}),
        warmup=execution.get("warmup"),
        iterations=execution.get("iterations"),
        latency=_latency_from_dict(metrics["latency"]),
        throughput_fps=metrics.get("throughput_fps"),
        power=_power_from_dict(metrics.get("power")),
        peak_memory_mb=metrics.get("peak_memory_mb"),
        peak_memory_scope=metrics.get("peak_memory_scope"),
        quality=document["quality"],
        source_id="autonomyfit-local",
        source_url="local://autonomyfit-benchmark",
        source_date=document["created_at"][:10],
        software_stack_id=None,
        notes=document.get("notes"),
        verified_identity=bool(model.get("revision") and artifact.get("sha256")),
    )


def _evidence_dir() -> Path:
    configured = os.environ.get("AUTONOMYFIT_EVIDENCE_DIR")
    if configured:
        return Path(configured).expanduser()
    return user_data_path("autonomyfit") / "evidence"


def local_benchmark_dir() -> Path:
    return _evidence_dir() / "benchmarks"


def load_local_benchmarks(
    directory: Path | None = None, hardware: Any | None = None
) -> list[BenchmarkEvidence]:
    root = directory or local_benchmark_dir()
    if not root.exists():
        return []
    benchmarks: list[BenchmarkEvidence] = []
    for path in sorted(root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                if hardware is not None:
                    from .local_results import local_report_compatibility

                    valid, _ = local_report_compatibility(value, hardware)
                    if not valid:
                        continue
                benchmarks.append(benchmark_evidence_from_report(value))
        except (OSError, json.JSONDecodeError, EvidenceError):
            continue
    return benchmarks


def import_benchmark_report(path: Path, directory: Path | None = None) -> Path:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceSchemaError(f"could not read benchmark report: {exc}") from exc
    if not isinstance(document, dict):
        raise EvidenceSchemaError("benchmark report must be a JSON object")
    validate_benchmark_report(document)
    target_dir = directory or local_benchmark_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{document['benchmark_id']}.json"
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    target.write_bytes(payload)
    return target


def load_evidence_store(
    include_local: bool = True, hardware: Any | None = None
) -> EvidenceStore:
    document = _load_json_resource("evidence-v2.json")
    validate_evidence_document(document)
    benchmarks = list(_bundled_benchmarks(document))
    if include_local:
        benchmarks.extend(load_local_benchmarks(hardware=hardware))
    benchmarks.sort(key=lambda item: (-QUALITY_RANK[item.quality], item.id))
    return EvidenceStore(document=document, benchmarks=tuple(benchmarks))


def _date_is_stale(value: str | None, max_age_days: int, today: date | None = None) -> bool:
    if not value:
        return True
    try:
        source_date = date.fromisoformat(value[:10])
    except ValueError:
        return True
    current = today or datetime.now(timezone.utc).date()
    return (current - source_date).days > max_age_days


def match_benchmarks(
    benchmarks: list[BenchmarkEvidence] | tuple[BenchmarkEvidence, ...],
    *,
    model_id: str,
    model_revision: str | None,
    hardware_id: str | None,
    runtime: str,
    precision: str,
    artifact_sha256: str | None = None,
    runtime_version: str | None = None,
    provider: str | None = None,
    max_age_days: int = 730,
    today: date | None = None,
) -> list[EvidenceMatch]:
    if not hardware_id:
        return []
    matches: list[EvidenceMatch] = []
    for evidence in benchmarks:
        if evidence.model_id != model_id:
            continue
        if evidence.hardware_id != hardware_id:
            continue
        if evidence.runtime.casefold() != runtime.casefold():
            continue
        if evidence.precision.casefold() != precision.casefold():
            continue
        if (
            model_revision
            and evidence.model_revision
            and evidence.model_revision != model_revision
        ):
            continue
        if (
            artifact_sha256
            and evidence.artifact_sha256
            and evidence.artifact_sha256.casefold() != artifact_sha256.casefold()
        ):
            continue
        if (
            runtime_version
            and evidence.runtime_version
            and evidence.runtime_version.casefold() != runtime_version.casefold()
        ):
            continue
        if provider and evidence.provider and evidence.provider.casefold() != provider.casefold():
            continue

        reasons: list[str] = []
        identity_complete = True
        if not model_revision or not evidence.model_revision:
            identity_complete = False
            reasons.append("model revision is not pinned on both sides")
        if artifact_sha256 is None or evidence.artifact_sha256 is None:
            identity_complete = False
            reasons.append("artifact SHA-256 is not pinned on both sides")
        if evidence.runtime_version and runtime_version is None:
            identity_complete = False
            reasons.append("target runtime version is unknown")
        if provider and not evidence.provider:
            identity_complete = False
            reasons.append("target execution provider is pinned but evidence provider is unknown")
        if _date_is_stale(evidence.source_date, max_age_days, today=today):
            reasons.append("evidence is older than the configured freshness window")

        exact = (
            identity_complete
            and evidence.verified_identity
            and not any("older" in reason for reason in reasons)
        )
        matches.append(
            EvidenceMatch(
                evidence=evidence,
                exact=exact,
                identity_complete=identity_complete,
                reasons=tuple(reasons),
            )
        )

    matches.sort(
        key=lambda item: (
            not item.exact,
            -QUALITY_RANK[item.evidence.quality],
            item.evidence.id,
        )
    )
    return matches


def inspect_benchmark_report(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceSchemaError(f"could not read benchmark report: {exc}") from exc
    if not isinstance(document, dict):
        raise EvidenceSchemaError("benchmark report must be a JSON object")
    validate_benchmark_report(document)
    evidence = benchmark_evidence_from_report(document)
    return {
        "benchmark_id": evidence.id,
        "model_id": evidence.model_id,
        "model_revision": evidence.model_revision,
        "artifact_sha256": evidence.artifact_sha256,
        "hardware_id": evidence.hardware_id,
        "runtime": evidence.runtime,
        "runtime_version": evidence.runtime_version,
        "provider": evidence.provider,
        "precision": evidence.precision,
        "latency_ms": evidence.latency_ms,
        "fps": evidence.fps,
        "power_mean_w": evidence.power.mean_w,
        "power_scope": evidence.power.scope,
        "peak_memory_mb": evidence.peak_memory_mb,
        "quality": evidence.quality,
        "identity_complete": evidence.eligible_for_verified_fit,
    }


def evidence_to_dict(evidence: BenchmarkEvidence) -> dict[str, Any]:
    value = asdict(evidence)
    value["latency_ms"] = evidence.latency_ms
    value["fps"] = evidence.fps
    value["eligible_for_verified_fit"] = evidence.eligible_for_verified_fit
    return value
