from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from .benchmark import hardware_evidence_id
from .catalog import load_model_catalog
from .evidence import (
    EvidenceMatch,
    EvidenceStore,
    load_evidence_store,
    match_benchmarks,
)
from .integrity import artifact_sha256
from .models import (
    ConfidenceBreakdown,
    Constraints,
    EvidenceConfidence,
    HardwareProfile,
    ModelProfile,
    Recommendation,
)
from .ranking import rank_recommendations
from .registry import RegistryClient
from .tasks import normalize_task

_BRIDGE_RUNTIMES = {
    "qnn": "QNNExecutionProvider",
    "xnnpack": "XNNPACKExecutionProvider",
    "openvino-ep": "OpenVINOExecutionProvider",
    "coreml-ep": "CoreMLExecutionProvider",
    "tensorrt-ep": "TensorrtExecutionProvider",
    "cuda-ep": "CUDAExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
}


def _runtime_available(hardware: HardwareProfile, runtime: str) -> bool:
    aliases = {
        "onnx": "onnxruntime",
        "onnxruntime": "onnxruntime",
        "pytorch": "pytorch",
        "tensorrt": "tensorrt",
        "coreml": "coreml",
        "openvino": "openvino",
        "transformers": "transformers",
    }
    capability_name = aliases.get(runtime, runtime)
    return any(cap.name == capability_name and cap.available for cap in hardware.runtimes)


def _runtime_capability(hardware: HardwareProfile, runtime: str):
    aliases = {"onnx": "onnxruntime", "onnxruntime": "onnxruntime"}
    name = aliases.get(runtime, runtime)
    return next((cap for cap in hardware.runtimes if cap.name == name), None)


def _runtime_version(hardware: HardwareProfile, runtime: str) -> str | None:
    capability = _runtime_capability(hardware, runtime)
    return capability.version if capability else None


def _runtime_compatibility(model: ModelProfile, runtime: str) -> tuple[bool, bool, str | None]:
    normalized = runtime.casefold()
    direct = {value.casefold() for value in model.runtimes}
    if normalized in direct or (normalized == "onnxruntime" and "onnx" in direct):
        return True, True, None
    if normalized in _BRIDGE_RUNTIMES and "onnx" in direct:
        return (
            True,
            False,
            f"{normalized} is an ONNX Runtime execution-provider path; model operator coverage is unverified",
        )
    return False, False, f"{runtime} is not listed for this model profile"


def choose_runtime(model: ModelProfile, hardware: HardwareProfile, requested: str | None) -> str:
    if requested:
        return requested.strip().casefold()
    runtimes = {value.casefold() for value in model.runtimes}
    if hardware.platform in {"jetson", "nvidia"} and "tensorrt" in runtimes:
        return "tensorrt"
    if hardware.platform == "apple" and "coreml" in runtimes and _runtime_available(hardware, "coreml"):
        return "coreml"
    if hardware.platform == "intel" and "openvino" in runtimes and _runtime_available(hardware, "openvino"):
        return "openvino"
    if hardware.platform == "qualcomm" and "qnn" in runtimes and _runtime_available(hardware, "qnn"):
        return "qnn"
    for candidate in ("onnx", "pytorch", "transformers"):
        if candidate in runtimes:
            return candidate
    return model.runtimes[0]


def choose_precision(model: ModelProfile, hardware: HardwareProfile, requested: str | None) -> str:
    if requested:
        return requested.strip().casefold()
    supported = {value.casefold() for value in model.supported_precisions}
    hardware_supported = {value.casefold() for value in hardware.supported_precisions}
    if hardware.accelerator_type != "cpu" and "fp16" in supported and (
        not hardware_supported or "fp16" in hardware_supported
    ):
        return "fp16"
    if "fp32" in supported or not supported:
        return "fp32"
    if hardware_supported:
        common = sorted(supported & hardware_supported)
        if common:
            return common[0]
    return min(supported)


def estimate_memory_gb(model: ModelProfile, precision: str) -> tuple[float | None, str]:
    if model.published_memory_gb is not None:
        return model.published_memory_gb, "published"
    if model.params_m is None:
        return None, "unknown"
    bytes_per_parameter = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "fp8": 1.0,
        "int8": 1.0,
        "int4": 0.5,
        "q4": 0.5,
    }.get(precision.casefold(), 2.0)
    weight_gib = model.params_m * 1_000_000 * bytes_per_parameter / (1024**3)
    if model.task in {"vlm", "asr"}:
        return max(1.0, weight_gib * 1.8 + 0.75), "metadata-derived estimate"
    if model.task in {"classification", "embedding"}:
        return max(0.5, weight_gib * 3.0 + 0.25), "metadata-derived estimate"
    return max(0.75, weight_gib * 8.0 + 0.5), "metadata-derived estimate"


def _available_accelerator_memory(hardware: HardwareProfile) -> float:
    if hardware.accelerator_memory_gb is not None:
        return hardware.accelerator_memory_gb
    return hardware.ram_available_gb


def _select_evidence(
    store: EvidenceStore,
    *,
    hardware: HardwareProfile,
    model: ModelProfile,
    model_revision: str | None,
    runtime: str,
    precision: str,
    artifact_sha256: str | None,
    provider_override: str | None = None,
    provider_version: str | None = None,
    quantization: str | None = None,
    batch_size: int | None = None,
    input_shapes: dict[str, list[int]] | None = None,
    power_mode: str | None = None,
    software_stack_id: str | None = None,
) -> EvidenceMatch | None:
    bridge_provider = _BRIDGE_RUNTIMES.get(runtime)
    provider = provider_override or bridge_provider
    runtime_candidates = ("onnxruntime",) if bridge_provider else (
        ("onnxruntime", "onnx") if runtime in {"onnx", "onnxruntime"} else (runtime,)
    )
    for runtime_alias in runtime_candidates:
        matches = match_benchmarks(
            store.benchmarks,
            model_id=model.id,
            model_revision=model_revision,
            hardware_id=hardware_evidence_id(hardware),
            runtime=runtime_alias,
            precision=precision,
            artifact_sha256=artifact_sha256,
            runtime_version=_runtime_version(hardware, runtime),
            provider=provider,
            provider_version=provider_version,
            quantization=quantization,
            batch_size=batch_size,
            input_shapes=input_shapes,
            power_mode=power_mode,
            software_stack_id=software_stack_id,
        )
        if matches:
            return matches[0]
        if hardware.os_name != "profile" and hardware.matched_profile:
            profile_matches = match_benchmarks(
                store.benchmarks,
                model_id=model.id,
                model_revision=model_revision,
                hardware_id=hardware.matched_profile,
                runtime=runtime_alias,
                precision=precision,
                artifact_sha256=artifact_sha256,
                runtime_version=_runtime_version(hardware, runtime),
                provider=provider,
            )
            if profile_matches:
                contextual = profile_matches[0]
                return EvidenceMatch(
                    evidence=contextual.evidence,
                    exact=False,
                    identity_complete=False,
                    reasons=("profile-level evidence is contextual for this detected machine", *contextual.reasons),
                )
    return None


def _freshness_score(source_date: str | None, fallback_date: str | None) -> float:
    raw = source_date or fallback_date
    if not raw:
        return 0.30
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            parsed = date.fromisoformat(raw[:10])
        except ValueError:
            return 0.30
    age = max(0, (datetime.now(timezone.utc).date() - parsed).days)
    if age <= 180:
        return 1.00
    if age <= 365:
        return 0.85
    if age <= 730:
        return 0.65
    return 0.40


def _confidence_level(score: float) -> EvidenceConfidence:
    if score >= 85:
        return "HIGH"
    if score >= 60:
        return "MEDIUM"
    if score >= 35:
        return "LOW"
    return "UNKNOWN"


def _confidence(
    *,
    hardware: HardwareProfile,
    model: ModelProfile,
    runtime: str,
    precision: str,
    runtime_direct: bool,
    match: EvidenceMatch | None,
    unresolved: list[str],
    known_quantities: int,
    target_quantities: int,
    registry_stale: bool,
) -> ConfidenceBreakdown:
    evidence = match.evidence if match else None
    if match and match.exact:
        hardware_match = 1.0
    elif hardware.matched_profile:
        hardware_match = 0.85
    else:
        hardware_match = 0.55

    capability = _runtime_capability(hardware, runtime)
    runtime_precision_match = 0.30
    if capability and capability.available:
        runtime_precision_match = 0.85 if runtime_direct and capability.verified else 0.55
    elif hardware.os_name == "profile" and runtime_direct:
        runtime_precision_match = 0.75
    if model.supported_precisions and precision not in {p.casefold() for p in model.supported_precisions}:
        runtime_precision_match = min(runtime_precision_match, 0.25)

    evidence_quality = 0.20
    if evidence:
        evidence_quality = {
            "local-measured": 1.00,
            "standardized": 0.95,
            "vendor-published": 0.75,
            "third-party-reproducible": 0.60,
            "metadata-estimate": 0.35,
        }.get(evidence.quality, 0.20)
    freshness = _freshness_score(evidence.source_date if evidence else None, model.last_verified)

    if match and match.exact and evidence and evidence.model_revision and evidence.artifact_sha256:
        revision_identity = 1.0
    elif model.source_revision:
        revision_identity = 0.65
    else:
        revision_identity = 0.30

    quantity_coverage = known_quantities / max(1, target_quantities)
    components = (
        hardware_match,
        runtime_precision_match,
        evidence_quality,
        freshness,
        revision_identity,
        quantity_coverage,
    )
    score = 100.0 * sum(components) / len(components)
    confidence_unresolved = list(unresolved)
    if registry_stale:
        confidence_unresolved.append("registry freshness")
    if confidence_unresolved:
        score = min(score, 55.0)
    score = round(max(0.0, min(100.0, score)), 1)
    return ConfidenceBreakdown(
        score=score,
        level=_confidence_level(score),
        hardware_match=round(hardware_match, 3),
        runtime_precision_match=round(runtime_precision_match, 3),
        evidence_quality=round(evidence_quality, 3),
        freshness=round(freshness, 3),
        revision_identity=round(revision_identity, 3),
        quantity_coverage=round(quantity_coverage, 3),
        unresolved_constraints=tuple(dict.fromkeys(confidence_unresolved)),
    )


def _model_filters(models: list[ModelProfile], constraints: Constraints) -> list[ModelProfile]:
    selected = models
    if constraints.model_id:
        selected = [item for item in selected if item.id.casefold() == constraints.model_id.casefold()]
    if constraints.family:
        family = constraints.family.casefold()
        selected = [item for item in selected if item.family.casefold() == family]
    if constraints.license_spdx:
        license_id = constraints.license_spdx.casefold()
        selected = [
            item for item in selected if item.license_spdx and item.license_spdx.casefold() == license_id
        ]
    if constraints.license_status:
        status = constraints.license_status.casefold()
        selected = [item for item in selected if item.license_status.casefold() == status]
    if constraints.verified_only:
        selected = [
            item
            for item in selected
            if item.verification_status in {"source_verified", "compatibility_verified", "benchmarked"}
        ]
    if not constraints.include_experimental:
        selected = [
            item
            for item in selected
            if not item.experimental and item.verification_status != "discovered"
        ]
    return selected


def recommend_models(
    hardware: HardwareProfile,
    constraints: Constraints,
    catalog_path: Path | None = None,
    *,
    offline: bool = False,
    force_registry_refresh: bool = False,
    registry_client: RegistryClient | None = None,
    evidence_store: EvidenceStore | None = None,
) -> list[Recommendation]:
    task = normalize_task(constraints.task)
    loaded = load_model_catalog(
        catalog_path,
        offline=offline,
        force_refresh=force_registry_refresh,
        client=registry_client,
    )
    models = _model_filters([model for model in loaded.models if model.task == task], constraints)
    store = evidence_store or load_evidence_store(include_local=True, hardware=hardware)
    recommendations: list[Recommendation] = []

    artifact_sha = constraints.artifact_sha256
    if artifact_sha is None and constraints.artifact_path is not None:
        artifact_sha = artifact_sha256(constraints.artifact_path)
    available_memory = _available_accelerator_memory(hardware)

    for model in models:
        runtime = choose_runtime(model, hardware, constraints.runtime)
        precision = choose_precision(model, hardware, constraints.precision)
        runtime_compatible, runtime_direct, runtime_note = _runtime_compatibility(model, runtime)
        estimated_memory, memory_evidence = estimate_memory_gb(model, precision)
        model_revision = constraints.model_revision or model.source_revision
        match = _select_evidence(
            store,
            hardware=hardware,
            model=model,
            model_revision=model_revision,
            runtime=runtime,
            precision=precision,
            artifact_sha256=artifact_sha,
            provider_override=constraints.provider,
            provider_version=constraints.provider_version,
            quantization=constraints.quantization,
            batch_size=constraints.batch_size,
            input_shapes=constraints.input_shapes,
            power_mode=constraints.power_mode,
            software_stack_id=constraints.software_stack_id,
        )
        benchmark = match.evidence if match else None
        exact_performance = bool(match and match.exact and benchmark.eligible_for_verified_fit)
        runtime_ready = _runtime_available(hardware, runtime)

        reasons: list[str] = []
        blockers: list[str] = []
        unknowns: list[str] = []
        measured: list[str] = []
        estimated: list[str] = []
        unresolved: list[str] = []
        performance_unknown = False
        power_unknown = False

        if loaded.provenance.stale:
            reasons.append(
                "registry data is stale; confidence is capped until a fresh signed registry is available"
            )
            unknowns.append("current registry freshness")

        if not runtime_compatible:
            blockers.append(runtime_note or f"{runtime} is not compatible")
        elif runtime_note:
            reasons.append(runtime_note)
            unknowns.append("execution-provider operator coverage")

        if model.supported_precisions and precision not in {
            value.casefold() for value in model.supported_precisions
        }:
            blockers.append(f"{precision} is not listed as a supported precision for this model")

        if estimated_memory is None:
            unknowns.append("deployment memory")
            if constraints.max_memory_gb is not None:
                unresolved.append("maximum memory")
        else:
            evidence_label = "published metadata" if memory_evidence == "published" else "metadata estimate"
            estimated.append(f"memory {estimated_memory:.2f} GB ({evidence_label})")
            if estimated_memory > available_memory:
                blockers.append(
                    f"memory screen needs {estimated_memory:.2f} GB; {available_memory:.2f} GB is available"
                )
            elif constraints.max_memory_gb is not None and estimated_memory > constraints.max_memory_gb:
                blockers.append(
                    f"memory {estimated_memory:.2f} GB exceeds requested {constraints.max_memory_gb:.2f} GB"
                )
            else:
                reasons.append(
                    f"memory screen fits with {available_memory - estimated_memory:.2f} GB hardware headroom"
                )

        if constraints.max_params_m is not None:
            if model.params_m is None:
                unknowns.append("parameter count")
                unresolved.append("maximum parameters")
            elif model.params_m > constraints.max_params_m:
                blockers.append(
                    f"{model.params_m:.1f}M parameters exceeds requested {constraints.max_params_m:.1f}M"
                )
            else:
                reasons.append(f"{model.params_m:.1f}M parameters meets the parameter limit")

        if constraints.min_accuracy is not None:
            if model.accuracy is None:
                unknowns.append("task accuracy")
                unresolved.append("accuracy threshold")
            else:
                metric = model.accuracy
                passed = (
                    metric.value >= constraints.min_accuracy
                    if metric.higher_is_better
                    else metric.value <= constraints.min_accuracy
                )
                relation = ">=" if metric.higher_is_better else "<="
                if not passed:
                    blockers.append(
                        f"{metric.name} {metric.value:.2f} does not meet {relation} {constraints.min_accuracy:.2f}"
                    )
                else:
                    reasons.append(
                        f"{metric.name} {metric.value:.2f} meets the direction-aware accuracy threshold"
                    )

        if benchmark:
            identity = "exact" if exact_performance else "context only"
            if benchmark.latency_ms is not None:
                measured.append(
                    f"latency {benchmark.latency_ms:.2f} ms ({benchmark.evidence_label}, {identity})"
                )
            if benchmark.fps is not None:
                measured.append(f"throughput {benchmark.fps:.1f} FPS ({benchmark.evidence_label}, {identity})")
            if benchmark.power.mean_w is not None:
                measured.append(
                    f"power {benchmark.power.mean_w:.1f} W ({benchmark.power.scope or 'scope unknown'})"
                )
            reasons.append(f"{benchmark.evidence_label} benchmark is {identity}")
            if match and match.reasons:
                reasons.append("evidence limitations: " + "; ".join(match.reasons))

        if constraints.min_fps is not None or constraints.max_latency_ms is not None:
            if not exact_performance or benchmark is None:
                performance_unknown = True
                unresolved.append("latency/throughput")
                unknowns.append("exact identity-matched latency/throughput")
            else:
                if constraints.min_fps is not None:
                    if benchmark.fps is None:
                        performance_unknown = True
                        unresolved.append("throughput")
                    elif benchmark.fps < constraints.min_fps:
                        blockers.append(
                            f"exact measured {benchmark.fps:.1f} FPS is below {constraints.min_fps:.1f} FPS"
                        )
                if constraints.max_latency_ms is not None:
                    if benchmark.latency_ms is None:
                        performance_unknown = True
                        unresolved.append("latency")
                    elif benchmark.latency_ms > constraints.max_latency_ms:
                        blockers.append(
                            f"exact measured {benchmark.latency_ms:.2f} ms exceeds {constraints.max_latency_ms:.2f} ms"
                        )
                if not blockers and not performance_unknown:
                    reasons.append("exact identity-matched performance evidence satisfies the constraint")

        if constraints.max_power_w is not None:
            if not exact_performance or benchmark is None or benchmark.power.mean_w is None:
                power_unknown = True
                unresolved.append("power")
                unknowns.append("exact scoped power")
            elif benchmark.power.mean_w > constraints.max_power_w:
                blockers.append(
                    f"exact measured {benchmark.power.mean_w:.1f} W exceeds {constraints.max_power_w:.1f} W"
                )
            else:
                reasons.append(
                    f"exact measured power {benchmark.power.mean_w:.1f} W meets the limit "
                    f"({benchmark.power.scope or 'scope unspecified'})"
                )

        if runtime_ready:
            capability = _runtime_capability(hardware, runtime)
            suffix = "verified hardware capability" if capability and capability.verified else "runtime available"
            reasons.append(f"{runtime} is available ({suffix})")
        else:
            reasons.append(f"{runtime} is a target runtime but is not currently installed")

        hard_fit_failure = any(
            marker in blocker
            for blocker in blockers
            for marker in (
                "memory screen needs",
                "exceeds requested",
                "not compatible",
                "not listed for this model",
                "not listed as a supported precision",
            )
        )
        if hard_fit_failure:
            verdict = "NO_FIT"
        elif blockers:
            verdict = "CONSTRAINT_FAIL"
        elif performance_unknown or power_unknown or unresolved:
            verdict = "BENCHMARK_REQUIRED"
        elif exact_performance:
            verdict = "VERIFIED_FIT"
        else:
            verdict = "FEASIBLE"

        requested_checks = []
        if constraints.min_fps is not None:
            requested_checks.append(exact_performance and benchmark is not None and benchmark.fps is not None)
        if constraints.max_latency_ms is not None:
            requested_checks.append(
                exact_performance and benchmark is not None and benchmark.latency_ms is not None
            )
        if constraints.max_power_w is not None:
            requested_checks.append(
                exact_performance
                and benchmark is not None
                and benchmark.power.mean_w is not None
            )
        if constraints.min_accuracy is not None:
            requested_checks.append(model.accuracy is not None)
        if constraints.max_memory_gb is not None:
            requested_checks.append(estimated_memory is not None)
        if constraints.max_params_m is not None:
            requested_checks.append(model.params_m is not None)

        if requested_checks:
            known_count = sum(requested_checks)
            target_count = len(requested_checks)
        else:
            known_count = sum(
                (
                    benchmark is not None and benchmark.latency_ms is not None,
                    benchmark is not None and benchmark.fps is not None,
                    model.accuracy is not None,
                    benchmark is not None and benchmark.power.mean_w is not None,
                    estimated_memory is not None,
                )
            )
            target_count = 5

        confidence = _confidence(
            hardware=hardware,
            model=model,
            runtime=runtime,
            precision=precision,
            runtime_direct=runtime_direct,
            match=match,
            unresolved=sorted(set(unresolved)),
            known_quantities=max(0, known_count),
            target_quantities=max(1, target_count),
            registry_stale=loaded.provenance.stale,
        )
        next_benchmark = None
        if unresolved:
            next_benchmark = (
                f"benchmark the exact {model.id} artifact on {hardware.matched_profile or hardware.platform} "
                f"with {runtime}/{precision} and import the JSON evidence"
            )
        recommendations.append(
            Recommendation(
                model=model,
                verdict=verdict,
                score=0.0,
                runtime=runtime,
                precision=precision,
                estimated_memory_gb=estimated_memory,
                memory_evidence=memory_evidence,
                benchmark=benchmark,
                evidence_match=match,
                evidence_confidence=confidence.level,
                runtime_available=runtime_ready,
                reasons=tuple(reasons),
                blockers=tuple(blockers),
                registry=loaded.provenance,
                confidence=confidence,
                unknowns=tuple(dict.fromkeys(unknowns)),
                measured=tuple(dict.fromkeys(measured)),
                estimated=tuple(dict.fromkeys(estimated)),
                next_benchmark=next_benchmark,
                objective=constraints.objective,
            )
        )

    ranked = rank_recommendations(recommendations, constraints.objective)
    if constraints.min_confidence is not None:
        ranked = [
            item
            for item in ranked
            if item.confidence is not None and item.confidence.score >= constraints.min_confidence
        ]
    return ranked
