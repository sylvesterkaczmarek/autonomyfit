from __future__ import annotations

import json
import shlex
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactError,
    ArtifactManager,
    ManagedArtifact,
    verify_artifact_identity,
)
from .backends import BackendError, BenchmarkRequest, backend_status, run_benchmark
from .catalog import load_model_catalog
from .conversions import (
    ConversionError,
    ConversionResult,
    compare_onnx_openvino_outputs,
    convert_artifact,
)
from .evidence import (
    EvidenceError,
    EvidenceStore,
    benchmark_evidence_from_report,
    import_benchmark_report,
    load_evidence_store,
    match_benchmarks,
)
from .hardware import detect_hardware, hardware_from_profile
from .integrity import artifact_members, artifact_size_bytes
from .models import Constraints, HardwareProfile, ModelProfile
from .ranking import rank_recommendations
from .reporting import recommendation_dict
from .scoring import choose_precision, choose_runtime, recommend_models

_BRIDGE_PROVIDERS = {
    "qnn": "QNNExecutionProvider",
    "xnnpack": "XNNPACKExecutionProvider",
    "openvino-ep": "OpenVINOExecutionProvider",
    "coreml-ep": "CoreMLExecutionProvider",
    "tensorrt-ep": "TensorrtExecutionProvider",
    "cuda-ep": "CUDAExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
}


class DeploymentValidationError(RuntimeError):
    """Deployment validation could not be completed safely."""


@dataclass(frozen=True)
class ValidationOptions:
    model_id: str
    artifact: Path | None = None
    artifact_url: str | None = None
    filename: str | None = None
    revision: str | None = None
    expected_sha256: str | None = None
    fetch: bool = False
    offline: bool = False
    runtime: str | None = None
    precision: str | None = None
    provider: str | None = None
    device: str | None = None
    convert: bool = False
    benchmark: bool = False
    iterations: int = 50
    warmup: int = 10
    shape: list[int] | None = None
    trust_artifact: bool = False
    import_local: bool = True
    hardware_profile: str | None = None
    max_latency_ms: float | None = None
    min_fps: float | None = None
    max_power_w: float | None = None
    max_memory_gb: float | None = None
    allow_restricted_license: bool = False
    compute_units: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if sum((self.artifact is not None, self.artifact_url is not None, bool(self.fetch))) > 1:
            raise ValueError("select only one artifact source")
        if self.offline and self.artifact_url is not None:
            raise ValueError("offline mode cannot acquire an artifact URL; select a local artifact")
        for name, minimum in (("iterations", 1), ("warmup", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.shape is not None and (
            not isinstance(self.shape, list) or not self.shape
            or any(type(size) is not int or size < 1 for size in self.shape)
        ):
            raise ValueError("shape must be a nonempty list of positive integers")
        Constraints(
            max_latency_ms=self.max_latency_ms, min_fps=self.min_fps,
            max_power_w=self.max_power_w, max_memory_gb=self.max_memory_gb,
        )


def _package_version() -> str:
    try:
        return version("autonomyfit")
    except PackageNotFoundError:
        return "source-tree"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _software_stack() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "autonomyfit_version": _package_version(),
        "benchmark_backends": [asdict(item) for item in backend_status()],
    }


def _constraints_payload(options: ValidationOptions) -> dict[str, Any]:
    return {
        "max_latency_ms": options.max_latency_ms,
        "min_fps": options.min_fps,
        "max_power_w": options.max_power_w,
        "max_memory_gb": options.max_memory_gb,
    }


def _find_model(model_id: str, offline: bool) -> tuple[ModelProfile, Any]:
    loaded = load_model_catalog(offline=offline)
    needle = model_id.casefold()
    model = next(
        (
            item
            for item in loaded.models
            if item.id.casefold() == needle or item.display_name.casefold() == needle
        ),
        None,
    )
    if model is None:
        raise DeploymentValidationError(f"unknown model: {model_id}")
    return model, loaded.provenance


def _resolve_hardware(profile: str | None) -> HardwareProfile:
    try:
        return hardware_from_profile(profile) if profile else detect_hardware()
    except ValueError as exc:
        raise DeploymentValidationError(str(exc)) from exc


def _runtime_capability(hardware: HardwareProfile, runtime: str) -> tuple[bool, str | None, str | None]:
    normalized = runtime.casefold()
    aliases = {"onnx": "onnxruntime", "onnxruntime": "onnxruntime"}
    name = aliases.get(normalized, normalized)
    for capability in hardware.runtimes:
        if capability.name.casefold() == name:
            return capability.available, capability.version, capability.detail
    if normalized in _BRIDGE_PROVIDERS:
        for capability in hardware.runtimes:
            if capability.name.casefold() == normalized:
                return capability.available, capability.version, capability.detail
    statuses = {item.name: item for item in backend_status()}
    backend_name = _benchmark_backend(runtime)
    status = statuses.get(backend_name)
    if status:
        return status.available, status.version, status.detail
    return False, None, "runtime was not detected"


def _model_runtime_check(model: ModelProfile, runtime: str) -> tuple[str, str]:
    normalized = runtime.casefold()
    direct = {value.casefold() for value in model.runtimes}
    if normalized in direct or (normalized == "onnxruntime" and "onnx" in direct):
        return "pass", f"{runtime} is directly listed for this model profile"
    if normalized in _BRIDGE_PROVIDERS and "onnx" in direct:
        return (
            "info",
            f"{runtime} is an ONNX Runtime provider path; provider presence is not proof of full graph coverage",
        )
    return "fail", f"{runtime} is not listed as a compatible runtime for this model profile"


def _benchmark_backend(runtime: str) -> str:
    normalized = runtime.casefold()
    if normalized in _BRIDGE_PROVIDERS:
        return "onnxruntime"
    if normalized in {"onnx", "onnxruntime"}:
        return "onnxruntime"
    return normalized


def _provider_for_runtime(runtime: str, explicit: str | None) -> str | None:
    return explicit or _BRIDGE_PROVIDERS.get(runtime.casefold())


def _check_onnx(path: Path) -> tuple[str, str]:
    try:
        import onnx
    except ImportError:
        return "skipped", "install autonomyfit[deployment] for ONNX structural validation"
    try:
        # Validate every external tensor path before asking a checker to read it.
        artifact_members(path)
        onnx.checker.check_model(str(path))
        return "pass", "ONNX checker accepted the model structure and external-data paths"
    except Exception as exc:  # noqa: BLE001
        return "fail", f"ONNX structural validation failed: {exc}"


def _check_safetensors(path: Path) -> tuple[str, str]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(8)
        if len(raw) != 8:
            return "fail", "safetensors file is too short"
        header_len = int.from_bytes(raw, "little")
        if header_len <= 0 or header_len > path.stat().st_size - 8 or header_len > 100 * 1024 * 1024:
            return "fail", "safetensors header length is invalid"
        with path.open("rb") as stream:
            stream.seek(8)
            header = json.loads(stream.read(header_len).decode("utf-8"))
        if not isinstance(header, dict):
            return "fail", "safetensors header is not a JSON object"
        return "pass", "safetensors header is structurally readable without pickle execution"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "fail", f"safetensors structural validation failed: {exc}"


def structural_checks(
    artifact: ManagedArtifact,
    *,
    runtime: str,
    runtime_available: bool,
    model: ModelProfile,
) -> tuple[list[dict[str, str]], list[str]]:
    checks: list[dict[str, str]] = []
    warnings: list[str] = []
    runtime_status, runtime_detail = _model_runtime_check(model, runtime)
    checks.append({"name": "model-runtime", "status": runtime_status, "detail": runtime_detail})
    checks.append(
        {
            "name": "runtime-availability",
            "status": "pass" if runtime_available else "fail",
            "detail": "runtime detected on this target" if runtime_available else "runtime is not currently available",
        }
    )
    if artifact.format == "onnx":
        status, detail = _check_onnx(artifact.path)
        checks.append({"name": "onnx-structure", "status": status, "detail": detail})
    elif artifact.format == "safetensors":
        status, detail = _check_safetensors(artifact.path)
        checks.append({"name": "safetensors-structure", "status": status, "detail": detail})
    elif artifact.format == "tensorrt-engine":
        if artifact.trusted_for_execution:
            checks.append(
                {
                    "name": "tensorrt-trust-boundary",
                    "status": "pass",
                    "detail": "engine is explicitly trusted or was built locally",
                }
            )
        else:
            checks.append(
                {
                    "name": "tensorrt-trust-boundary",
                    "status": "fail",
                    "detail": "untrusted TensorRT engines are executable artifacts and will not be deserialized",
                }
            )
    else:
        checks.append(
            {
                "name": "artifact-format",
                "status": "info",
                "detail": f"no generic structural validator is available for {artifact.format}",
            }
        )
    if artifact.remote_code_required and artifact.format not in {"onnx", "safetensors"}:
        checks.append(
            {
                "name": "remote-code",
                "status": "fail",
                "detail": "upstream repository requires custom code and no self-contained safe artifact was selected",
            }
        )
    elif artifact.remote_code_required:
        warnings.append(
            "upstream repository contains custom/remote code; AutonomyFit did not execute it and only handles the selected static artifact"
        )
    return checks, warnings


def _registry_comparison(
    *,
    model: ModelProfile,
    hardware: HardwareProfile,
    runtime: str,
    precision: str,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    from .benchmark import hardware_evidence_id

    store = load_evidence_store(include_local=False)
    provider = _provider_for_runtime(runtime, None)
    runtime_alias = "onnxruntime" if runtime.casefold() in _BRIDGE_PROVIDERS else (
        "onnx" if runtime in {"onnx", "onnxruntime"} else runtime
    )
    matches = match_benchmarks(
        store.benchmarks,
        model_id=model.id,
        model_revision=model.source_revision,
        hardware_id=hardware_evidence_id(hardware),
        runtime=runtime_alias,
        precision=precision,
        provider=provider,
    )
    if not matches and model.source_revision:
        matches = match_benchmarks(
            store.benchmarks,
            model_id=model.id,
            model_revision=None,
            hardware_id=hardware_evidence_id(hardware),
            runtime=runtime_alias,
            precision=precision,
            provider=provider,
        )
    local_metrics = benchmark.get("metrics") or {}
    local_latency = (local_metrics.get("latency") or {}).get("median_ms") or (
        local_metrics.get("latency") or {}
    ).get("mean_ms")
    if not matches:
        return {
            "classification": "no-comparable-registry-evidence",
            "local_latency_ms": local_latency,
            "expected_latency_ms": None,
            "expected_range_ms": None,
            "ratio": None,
            "warnings": ["no same-hardware/runtime/precision registry benchmark was available"],
        }
    evidence = matches[0].evidence
    warnings: list[str] = []
    mismatches: list[str] = []
    execution = benchmark.get("execution") or {}
    if evidence.batch_size and execution.get("batch_size") and evidence.batch_size != execution.get("batch_size"):
        mismatches.append("batch size differs from registry evidence")
    local_shapes = execution.get("input_shapes") or {}
    if evidence.input_shapes and local_shapes and evidence.input_shapes != local_shapes:
        mismatches.append("input shapes differ from registry evidence")
    local_power_mode = (benchmark.get("hardware") or {}).get("power_mode")
    if evidence.power_mode and local_power_mode and evidence.power_mode != local_power_mode:
        mismatches.append("power mode differs from registry evidence")
    registry_latency = evidence.latency_ms
    lower = evidence.latency.p50_ms or evidence.latency.median_ms or evidence.latency.mean_ms
    upper = evidence.latency.p95_ms or evidence.latency.p99_ms or evidence.latency.max_ms
    expected_range = (
        [round(float(lower), 4), round(float(upper), 4)]
        if lower is not None and upper is not None and float(upper) >= float(lower)
        else None
    )
    warnings.append(
        "registry benchmark evidence does not encode a directly comparable thermal state; local thermals are recorded separately"
    )
    if evidence.software_stack_id:
        mismatches.append(
            f"registry evidence references software stack {evidence.software_stack_id}; exact stack equivalence must be checked before attributing the difference"
        )
    if mismatches or local_latency is None or registry_latency is None:
        classification = "not-comparable"
        ratio = None
    else:
        ratio = float(local_latency) / float(registry_latency)
        if ratio > 1.20:
            classification = "materially-slower"
        elif ratio < 0.80:
            classification = "materially-faster"
        else:
            classification = "within-20-percent"
    warnings = mismatches + warnings
    return {
        "classification": classification,
        "local_latency_ms": local_latency,
        "expected_latency_ms": registry_latency,
        "expected_range_ms": expected_range,
        "ratio": ratio,
        "registry_evidence_id": evidence.id,
        "registry_quality": evidence.quality,
        "warnings": warnings,
        "note": "20% is an engineering comparison threshold, not a statistical significance test",
    }


def _recommendation_after_local_measurement(
    *,
    model: ModelProfile,
    hardware: HardwareProfile,
    artifact: ManagedArtifact,
    runtime: str,
    precision: str,
    options: ValidationOptions,
    benchmark: dict[str, Any],
) -> dict[str, Any] | None:
    software = benchmark.get("software") or {}
    execution = benchmark.get("execution") or {}
    benchmark_hardware = benchmark.get("hardware") or {}
    reproducibility = benchmark.get("reproducibility") or {}
    constraints = Constraints(
        task=model.task,
        model_id=model.id,
        model_revision=artifact.resolved_revision or options.revision or model.source_revision,
        artifact_sha256=artifact.sha256,
        runtime=runtime,
        precision=precision,
        max_latency_ms=options.max_latency_ms,
        min_fps=options.min_fps,
        max_power_w=options.max_power_w,
        max_memory_gb=options.max_memory_gb,
        include_experimental=True,
        provider=software.get("provider"),
        provider_version=software.get("provider_version"),
        quantization=execution.get("quantization"),
        batch_size=execution.get("batch_size"),
        input_shapes=execution.get("input_shapes") or {},
        power_mode=benchmark_hardware.get("power_mode"),
        software_stack_id=reproducibility.get("software_stack_fingerprint"),
    )
    # The result of this run must drive the assessment even when persistence is
    # disabled or a previous measurement of the same artifact is cached.
    fresh = benchmark_evidence_from_report(benchmark)
    items = recommend_models(
        hardware, constraints, offline=options.offline,
        evidence_store=EvidenceStore(document={}, benchmarks=(fresh,)),
    )
    if not items:
        return None
    return recommendation_dict(items[0])


def _managed_from_conversion(
    model: ModelProfile,
    source: ManagedArtifact,
    conversion: ConversionResult,
) -> ManagedArtifact:
    return ManagedArtifact(
        model_id=model.id,
        path=conversion.target_path,
        format=conversion.target_format,
        sha256=conversion.target_sha256,
        size_bytes=artifact_size_bytes(conversion.target_path),
        source="local-conversion",
        provenance_url=source.provenance_url,
        requested_revision=source.requested_revision,
        resolved_revision=source.resolved_revision,
        filename=conversion.target_path.name,
        license_spdx=model.license_spdx,
        license_status=model.license_status,
        remote_code_required=False,
        trusted_for_execution=True,
        cached=False,
        acquired_at=_now(),
    )


def _benchmark_command(
    model: ModelProfile, artifact: ManagedArtifact, options: ValidationOptions,
    runtime: str, precision: str, provider: str | None,
) -> str:
    args = [
        "autonomyfit", "validate", model.id, "--artifact", str(artifact.path),
        "--sha256", artifact.sha256, "--runtime", runtime, "--precision", precision,
        "--benchmark", "--iterations", str(options.iterations), "--warmup", str(options.warmup),
    ]
    values = {
        "--revision": artifact.resolved_revision or options.revision or model.source_revision,
        "--provider": provider, "--device": options.device, "--compute-units": options.compute_units,
        "--shape": ",".join(map(str, options.shape)) if options.shape else None,
        "--latency-ms": options.max_latency_ms, "--fps": options.min_fps,
        "--power-w": options.max_power_w, "--max-memory-gb": options.max_memory_gb,
    }
    for flag, value in values.items():
        if value is not None:
            args.extend((flag, str(value)))
    if artifact.trusted_for_execution:
        args.append("--trust-artifact")
    if options.offline:
        args.append("--offline")
    if not options.import_local:
        args.append("--no-import-local")
    return shlex.join(args)


def validate_deployment(options: ValidationOptions) -> dict[str, Any]:
    model, provenance = _find_model(options.model_id, options.offline)
    hardware = _resolve_hardware(options.hardware_profile)
    if options.benchmark and options.hardware_profile:
        actual = detect_hardware()
        if actual.matched_profile != options.hardware_profile:
            raise DeploymentValidationError(
                "--benchmark measures the actual machine; the requested --hardware-profile does not match "
                f"the detected profile ({actual.matched_profile or 'unmatched'})"
            )
        hardware = actual
    runtime = choose_runtime(model, hardware, options.runtime)
    precision = choose_precision(model, hardware, options.precision)
    available, runtime_version, runtime_detail = _runtime_capability(hardware, runtime)
    manager = ArtifactManager()
    warnings: list[str] = []
    if model.license_status != "published":
        warnings.append(
            f"model licence status is {model.license_status}; AutonomyFit reports provenance but does not grant usage rights"
        )
    reproducibility: list[str] = ["autonomyfit scan"]

    if options.artifact is None and options.artifact_url is None and not options.fetch:
        discovery = None
        if model.source_url.startswith("https://huggingface.co/"):
            try:
                discovery = manager.discover_huggingface(
                    model, revision=options.revision, offline=options.offline
                )
            except ArtifactError as exc:
                warnings.append(str(exc))
        return {
            "schema_version": 1,
            "created_at": _now(),
            "status": "artifact-selection-required",
            "validation_scope": "identity-and-compatibility",
            "autonomyfit_version": _package_version(),
            "registry": provenance.to_dict(),
            "machine": hardware.to_dict(),
            "software_stack": _software_stack(),
            "constraints": _constraints_payload(options),
            "model": {
                "id": model.id,
                "display_name": model.display_name,
                "revision": options.revision or model.source_revision,
                "source_url": model.source_url,
                "license_spdx": model.license_spdx,
                "license_status": model.license_status,
                "license_source_url": model.license_source_url,
            },
            "artifact": None,
            "artifact_discovery": discovery,
            "runtime": {
                "name": runtime,
                "version": runtime_version,
                "detail": runtime_detail,
                "precision": precision,
                "available": available,
            },
            "conversion": None,
            "compatibility": {
                "checks": [
                    {
                        "name": "artifact-identity",
                        "status": "info",
                        "detail": "select a local artifact or use --fetch/--artifact-url before runtime validation",
                    }
                ]
            },
            "benchmark": None,
            "registry_comparison": None,
            "recommendation": None,
            "warnings": warnings,
            "reproducibility": {"commands": reproducibility},
        }

    try:
        if options.artifact is not None:
            managed = manager.manage_local(
                model,
                options.artifact,
                expected_sha256=options.expected_sha256,
                trusted_for_execution=options.trust_artifact,
                revision=options.revision,
            )
            reproducibility.append(
                f"autonomyfit validate {shlex.quote(model.id)} --artifact {shlex.quote(str(options.artifact))} --sha256 {managed.sha256} --runtime {shlex.quote(runtime)}"
            )
        elif options.artifact_url is not None:
            if not options.expected_sha256:
                raise DeploymentValidationError("--artifact-url requires --sha256")
            managed = manager.acquire_url(
                model,
                url=options.artifact_url,
                expected_sha256=options.expected_sha256,
                filename=options.filename,
                revision=options.revision,
                allow_restricted_license=options.allow_restricted_license,
            )
            reproducibility.append(
                f"autonomyfit validate {shlex.quote(model.id)} --artifact-url {shlex.quote(options.artifact_url)} --sha256 {managed.sha256} --runtime {shlex.quote(runtime)}"
            )
        else:
            managed = manager.acquire_huggingface(
                model,
                filename=options.filename,
                revision=options.revision,
                expected_sha256=options.expected_sha256,
                offline=options.offline,
                allow_restricted_license=options.allow_restricted_license,
            )
            reproducibility.append(
                f"autonomyfit validate {shlex.quote(model.id)} --fetch --filename {shlex.quote(managed.filename)} --revision {managed.resolved_revision} --sha256 {managed.sha256} --runtime {shlex.quote(runtime)}"
            )
    except ArtifactError as exc:
        raise DeploymentValidationError(str(exc)) from exc

    try:
        verify_artifact_identity(managed)
    except ArtifactError as exc:
        raise DeploymentValidationError(str(exc)) from exc

    checks, artifact_warnings = structural_checks(
        managed, runtime=runtime, runtime_available=available, model=model
    )
    warnings.extend(artifact_warnings)

    conversion_payload = None
    final_artifact = managed
    if options.convert:
        try:
            out_dir = manager.cache_dir / model.id / "conversions" / managed.sha256[:20]
            conversion = convert_artifact(
                managed.path,
                runtime,
                out_dir,
                precision=precision,
                input_shape=options.shape,
                input_shapes=({"input": options.shape} if options.shape else None),
                trust_source=options.trust_artifact,
                expected_source_sha256=managed.sha256,
            )
            if conversion.target_runtime == "openvino" and managed.format == "onnx":
                equivalence = compare_onnx_openvino_outputs(
                    managed.path, conversion.target_path, shape_override=options.shape
                )
                conversion = replace(conversion, equivalence=equivalence)
                if equivalence.get("status") == "failed":
                    checks.append({
                        "name": "conversion-equivalence", "status": "fail",
                        "detail": "converted outputs failed the numerical comparison; benchmark was not attempted",
                    })
                if equivalence.get("status") != "passed":
                    warnings.append(
                        "conversion completed but generic numerical equivalence was not established"
                    )
            else:
                warnings.append(
                    "conversion completed; task-level accuracy equivalence is not implied by conversion success"
                )
            conversion_payload = conversion.to_dict()
            if conversion.command:
                reproducibility.append(" ".join(shlex.quote(part) for part in conversion.command))
            final_artifact = _managed_from_conversion(model, managed, conversion)
            checks.append(
                {
                    "name": "conversion",
                    "status": "pass",
                    "detail": f"locally converted {managed.format} to {conversion.target_format} with {conversion.tool}",
                }
            )
        except ConversionError as exc:
            checks.append({"name": "conversion", "status": "fail", "detail": str(exc)})
            warnings.append("requested conversion failed; benchmark was not attempted on a converted artifact")

    failed = any(item["status"] == "fail" for item in checks)
    benchmark_report: dict[str, Any] | None = None
    comparison = None
    recommendation = None
    if options.benchmark and not failed:
        backend_name = _benchmark_backend(runtime)
        provider = _provider_for_runtime(runtime, options.provider)
        if final_artifact.format == "safetensors":
            checks.append(
                {
                    "name": "benchmark",
                    "status": "fail",
                    "detail": "safetensors contains weights only; a self-contained executable graph/export is required for generic benchmarking",
                }
            )
            failed = True
        else:
            command = _benchmark_command(model, final_artifact, options, runtime, precision, provider)
            request = BenchmarkRequest(
                model_path=final_artifact.path,
                model_id=model.id,
                model_revision=final_artifact.resolved_revision or options.revision or model.source_revision,
                hardware=hardware,
                iterations=options.iterations,
                warmup=options.warmup,
                shape_override=options.shape,
                provider=provider,
                device=options.device,
                precision=precision,
                command=command,
                trusted_artifact=final_artifact.trusted_for_execution,
                expected_sha256=final_artifact.sha256,
                compute_units=options.compute_units,
            )
            try:
                measured_report = run_benchmark(request, backend_name)
                benchmark_evidence_from_report(measured_report)
                benchmark_report = measured_report
            except (BackendError, EvidenceError, ValueError, RuntimeError) as exc:
                checks.append({"name": "benchmark", "status": "fail", "detail": str(exc)})
                failed = True
            else:
                checks.append(
                    {
                        "name": "benchmark",
                        "status": "pass",
                        "detail": f"measured {options.iterations} timed iterations on the current machine",
                    }
                )
                comparison = _registry_comparison(
                    model=model,
                    hardware=hardware,
                    runtime=runtime,
                    precision=precision,
                    benchmark=benchmark_report,
                )
                recommendation = _recommendation_after_local_measurement(
                    model=model,
                    hardware=hardware,
                    artifact=final_artifact,
                    runtime=runtime,
                    precision=precision,
                    options=options,
                    benchmark=benchmark_report,
                )
                if options.import_local:
                    try:
                        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
                            temp_path = Path(stream.name)
                            try:
                                json.dump(benchmark_report, stream, allow_nan=False)
                            except BaseException:
                                temp_path.unlink(missing_ok=True)
                                raise
                        try:
                            imported = import_benchmark_report(temp_path)
                            benchmark_report["local_evidence_path"] = str(imported)
                        finally:
                            temp_path.unlink(missing_ok=True)
                    except (OSError, EvidenceError) as exc:
                        warnings.append(f"benchmark completed but local evidence could not be saved: {exc}")
                reproducibility.append(command)

    failed_constraints = bool(recommendation and recommendation.get("blockers"))
    requested_perf = any(
        value is not None
        for value in (
            options.max_latency_ms,
            options.min_fps,
            options.max_power_w,
            options.max_memory_gb,
        )
    )
    if failed or failed_constraints:
        status = "constraint-fail" if failed_constraints and not failed else "failed"
    elif requested_perf and (
        benchmark_report is None or recommendation is None
        or recommendation.get("verdict") != "VERIFIED_FIT"
    ):
        status = "benchmark-required"
    else:
        status = "validated"
    if benchmark_report is None:
        warnings.append("deployment performance has not been measured on this machine")

    return {
        "schema_version": 1,
        "created_at": _now(),
        "status": status,
        "validation_scope": "measured-deployment" if benchmark_report is not None else "identity-and-compatibility",
        "autonomyfit_version": _package_version(),
        "registry": provenance.to_dict(),
        "machine": hardware.to_dict(),
        "software_stack": _software_stack(),
        "constraints": _constraints_payload(options),
        "model": {
            "id": model.id,
            "display_name": model.display_name,
            "revision": final_artifact.resolved_revision or options.revision or model.source_revision,
            "source_url": model.source_url,
            "license_spdx": model.license_spdx,
            "license_status": model.license_status,
            "license_source_url": model.license_source_url,
        },
        "artifact": final_artifact.to_dict(),
        "source_artifact": managed.to_dict(),
        "runtime": {
            "name": runtime,
            "version": runtime_version,
            "detail": runtime_detail,
            "precision": precision,
            "provider": _provider_for_runtime(runtime, options.provider),
            "available": available,
        },
        "conversion": conversion_payload,
        "compatibility": {"checks": checks},
        "benchmark": benchmark_report,
        "registry_comparison": comparison,
        "recommendation": recommendation,
        "warnings": list(dict.fromkeys(warnings)),
        "reproducibility": {"commands": reproducibility},
    }


def assess_candidates(
    model_ids: list[str],
    artifact_map: dict[str, Path],
    *,
    runtime: str | None = None,
    precision: str | None = None,
    iterations: int = 50,
    warmup: int = 10,
    offline: bool = False,
    hardware_profile: str | None = None,
) -> dict[str, Any]:
    if len(model_ids) < 2:
        raise DeploymentValidationError("candidate assessment requires at least two models")
    loaded = load_model_catalog(offline=offline)

    def resolve_model(name: str) -> ModelProfile:
        needle = name.strip().casefold()
        matches = [item for item in loaded.models if item.id.casefold() == needle]
        if not matches:
            matches = [item for item in loaded.models if item.display_name.casefold() == needle]
        if len(matches) != 1:
            raise DeploymentValidationError(f"unknown or ambiguous candidate model: {name}")
        return matches[0]

    selected = [resolve_model(model_id) for model_id in model_ids]
    selected_ids = {item.id.casefold() for item in selected}
    if len(selected_ids) != len(selected):
        raise DeploymentValidationError("candidate assessment requires unique model IDs")
    tasks = {item.task for item in selected}
    if len(tasks) != 1:
        raise DeploymentValidationError("candidate assessment requires models from one task")
    artifacts = {}
    for name, artifact in artifact_map.items():
        model = resolve_model(name)
        if model.id.casefold() not in selected_ids:
            raise DeploymentValidationError(f"artifact mapping is not a selected candidate: {name}")
        if model.id in artifacts:
            raise DeploymentValidationError(f"duplicate artifact mapping for candidate {model.id}")
        if artifact is None:
            raise DeploymentValidationError(f"candidate {model.id} has no artifact path")
        path = Path(artifact).expanduser()
        if not path.is_file() and not (path.is_dir() and path.suffix.casefold() == ".mlpackage"):
            raise DeploymentValidationError(f"candidate {model.id} artifact does not exist: {path}")
        artifacts[model.id] = path
    options = []
    for model in selected:
        if model.id not in artifacts:
            raise DeploymentValidationError(
                f"candidate {model.id} is missing an artifact mapping; use --artifact {model.id}=PATH"
            )
        options.append(
            ValidationOptions(
                model_id=model.id,
                artifact=artifacts[model.id],
                runtime=runtime,
                precision=precision,
                benchmark=True,
                iterations=iterations,
                warmup=warmup,
                offline=offline,
                hardware_profile=hardware_profile,
            )
        )
    hardware = detect_hardware()
    if hardware_profile and hardware.matched_profile != hardware_profile:
        raise DeploymentValidationError(
            "candidate assessment requires the requested hardware profile to match the actual machine"
        )
    reports = [validate_deployment(item) for item in options]
    chosen = []
    for model, report in zip(selected, reports):
        if report.get("status") in {"failed", "constraint-fail"} or not report.get("benchmark"):
            continue
        artifact = report.get("artifact") or {}
        # Persistence metadata is not part of the benchmark report schema.
        benchmark = {
            key: value for key, value in report["benchmark"].items()
            if key != "local_evidence_path"
        }
        try:
            fresh = benchmark_evidence_from_report(benchmark)
        except EvidenceError:
            continue
        if (
            fresh.model_id != model.id
            or fresh.artifact_sha256 != artifact.get("sha256")
            or (report.get("model") or {}).get("id") != model.id
        ):
            continue
        software = benchmark.get("software") or {}
        execution = benchmark.get("execution") or {}
        benchmark_hardware = benchmark.get("hardware") or {}
        reproducibility = benchmark.get("reproducibility") or {}
        exact = recommend_models(
            hardware,
            Constraints(
                task=next(iter(tasks)),
                model_id=model.id,
                model_revision=(report.get("model") or {}).get("revision"),
                artifact_sha256=artifact.get("sha256"),
                runtime=software.get("runtime"),
                precision=execution.get("precision"),
                include_experimental=True,
                provider=software.get("provider"),
                provider_version=software.get("provider_version"),
                quantization=execution.get("quantization"),
                batch_size=execution.get("batch_size"),
                input_shapes=execution.get("input_shapes") or {},
                power_mode=benchmark_hardware.get("power_mode"),
                software_stack_id=reproducibility.get("software_stack_fingerprint"),
            ),
            offline=offline,
            evidence_store=EvidenceStore(document={}, benchmarks=(fresh,)),
        )
        if exact and exact[0].benchmark is not None and exact[0].benchmark.id == fresh.id:
            chosen.append(exact[0])
    chosen = rank_recommendations(chosen, "balanced")
    return {
        "hardware": hardware.to_dict(),
        "reports": reports,
        "reordered_recommendations": [recommendation_dict(item) for item in chosen],
    }
