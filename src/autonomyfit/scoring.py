from __future__ import annotations

from pathlib import Path

from .benchmark import hardware_evidence_id
from .catalog import load_model_catalog
from .evidence import (
    EvidenceMatch,
    EvidenceStore,
    load_evidence_store,
    match_benchmarks,
    sha256_file,
)
from .models import Constraints, HardwareProfile, ModelProfile, Recommendation
from .registry import RegistryClient


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



def _runtime_version(hardware: HardwareProfile, runtime: str) -> str | None:
    aliases = {"onnx": "onnxruntime", "onnxruntime": "onnxruntime"}
    name = aliases.get(runtime, runtime)
    capability = next((cap for cap in hardware.runtimes if cap.name == name), None)
    return capability.version if capability else None


def choose_runtime(model: ModelProfile, hardware: HardwareProfile, requested: str | None) -> str:
    if requested:
        return requested.strip().lower()
    if hardware.platform in {"jetson", "nvidia"} and "tensorrt" in model.runtimes:
        return "tensorrt"
    if hardware.platform == "apple":
        if "coreml" in model.runtimes and _runtime_available(hardware, "coreml"):
            return "coreml"
        if "pytorch" in model.runtimes:
            return "pytorch"
    if "onnx" in model.runtimes:
        return "onnx"
    if "pytorch" in model.runtimes:
        return "pytorch"
    if "transformers" in model.runtimes:
        return "transformers"
    return model.runtimes[0]


def choose_precision(model: ModelProfile, hardware: HardwareProfile, requested: str | None) -> str:
    if requested:
        return requested.strip().lower()
    if model.task == "vlm":
        return "fp16" if hardware.platform != "cpu" else "fp32"
    if hardware.platform in {"jetson", "nvidia", "apple"}:
        return "fp16"
    return "fp32"


def estimate_memory_gb(model: ModelProfile, precision: str) -> tuple[float, str]:
    if model.published_memory_gb is not None:
        return model.published_memory_gb, "published"
    bytes_per_parameter = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
        "q4": 0.5,
    }.get(precision.lower(), 2.0)
    weight_gib = model.params_m * 1_000_000 * bytes_per_parameter / (1024**3)
    if model.task == "detection":
        return max(1.0, weight_gib * 16.0 + 0.75), "metadata-derived estimate"
    return max(1.5, weight_gib * 1.6 + 1.0), "metadata-derived estimate"


def _available_accelerator_memory(hardware: HardwareProfile) -> float:
    if hardware.accelerator_memory_gb is not None:
        return hardware.accelerator_memory_gb
    return hardware.ram_available_gb


def _confidence(match: EvidenceMatch | None) -> str:
    if match is None:
        return "UNKNOWN"
    if match.exact and match.evidence.eligible_for_verified_fit:
        return "HIGH"
    if match.evidence.quality in {"standardized", "vendor-published", "local-measured"}:
        return "MEDIUM"
    if match.evidence.quality in {"third-party-reproducible", "metadata-estimate"}:
        return "LOW"
    return "UNKNOWN"


def _select_evidence(
    store: EvidenceStore,
    *,
    hardware: HardwareProfile,
    model: ModelProfile,
    model_revision: str | None,
    runtime: str,
    precision: str,
    artifact_sha256: str | None,
) -> EvidenceMatch | None:
    runtime_alias = "onnx" if runtime == "onnxruntime" else runtime
    matches = match_benchmarks(
        store.benchmarks,
        model_id=model.id,
        model_revision=model_revision,
        hardware_id=hardware_evidence_id(hardware),
        runtime=runtime_alias,
        precision=precision,
        artifact_sha256=artifact_sha256,
        runtime_version=_runtime_version(hardware, runtime),
    )
    return matches[0] if matches else None


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
    loaded = load_model_catalog(
        catalog_path,
        offline=offline,
        force_refresh=force_registry_refresh,
        client=registry_client,
    )
    models = [model for model in loaded.models if model.task == constraints.task]
    if constraints.model_id:
        models = [model for model in models if model.id.casefold() == constraints.model_id.casefold()]
    store = evidence_store or load_evidence_store(include_local=True)
    recommendations: list[Recommendation] = []

    artifact_sha = constraints.artifact_sha256
    if artifact_sha is None and constraints.artifact_path is not None:
        artifact_sha = sha256_file(constraints.artifact_path)

    task_accuracy = [model.accuracy.value for model in models if model.accuracy is not None]
    min_acc = min(task_accuracy) if task_accuracy else 0.0
    max_acc = max(task_accuracy) if task_accuracy else 1.0
    acc_span = max(max_acc - min_acc, 1e-9)
    available_memory = _available_accelerator_memory(hardware)

    for model in models:
        runtime = choose_runtime(model, hardware, constraints.runtime)
        precision = choose_precision(model, hardware, constraints.precision)
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
        )
        benchmark = match.evidence if match else None
        exact_performance = bool(match and match.exact and benchmark.eligible_for_verified_fit)
        runtime_ready = _runtime_available(hardware, runtime)

        reasons: list[str] = []
        blockers: list[str] = []
        performance_unknown = False
        power_unknown = False

        if runtime not in model.runtimes and not (runtime == "onnxruntime" and "onnx" in model.runtimes):
            blockers.append(f"{runtime} is not listed for this model profile")

        if estimated_memory > available_memory:
            blockers.append(
                f"memory screen needs {estimated_memory:.2f} GB; {available_memory:.2f} GB is available"
            )
        else:
            reasons.append(
                f"memory screen fits with {available_memory - estimated_memory:.2f} GB headroom"
            )

        if constraints.min_accuracy is not None:
            if model.accuracy is None:
                blockers.append("accuracy constraint cannot be evaluated for this profile")
            elif model.accuracy.value < constraints.min_accuracy:
                blockers.append(
                    f"{model.accuracy.name} {model.accuracy.value:.2f} is below {constraints.min_accuracy:.2f}"
                )
            else:
                reasons.append(
                    f"{model.accuracy.name} {model.accuracy.value:.2f} meets the accuracy constraint"
                )

        if benchmark:
            identity = "exact" if exact_performance else "context only"
            reasons.append(
                f"{benchmark.evidence_label} benchmark ({identity}): "
                f"{benchmark.latency_ms:.2f} ms" if benchmark.latency_ms is not None
                else f"{benchmark.evidence_label} benchmark ({identity}) has no latency metric"
            )
            if match and match.reasons:
                reasons.append("evidence limitations: " + "; ".join(match.reasons))

        if constraints.min_fps is not None or constraints.max_latency_ms is not None:
            if not exact_performance or benchmark is None:
                performance_unknown = True
            else:
                if constraints.min_fps is not None:
                    if benchmark.fps is None:
                        performance_unknown = True
                    elif benchmark.fps < constraints.min_fps:
                        blockers.append(
                            f"exact measured {benchmark.fps:.1f} FPS is below {constraints.min_fps:.1f} FPS"
                        )
                if constraints.max_latency_ms is not None:
                    if benchmark.latency_ms is None:
                        performance_unknown = True
                    elif benchmark.latency_ms > constraints.max_latency_ms:
                        blockers.append(
                            f"exact measured {benchmark.latency_ms:.2f} ms exceeds "
                            f"{constraints.max_latency_ms:.2f} ms"
                        )
                if not blockers and not performance_unknown:
                    reasons.append("exact identity-matched performance evidence satisfies the constraint")

        if constraints.max_power_w is not None:
            if not exact_performance or benchmark is None or benchmark.power.mean_w is None:
                power_unknown = True
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
            reasons.append(
                f"{runtime} is supported by the selected hardware profile"
                if hardware.os_name == "profile"
                else f"{runtime} is available on this environment"
            )
        else:
            reasons.append(f"{runtime} is a target runtime but is not currently installed")

        hard_memory_failure = any("memory screen needs" in blocker for blocker in blockers)
        runtime_failure = any("not listed for this model" in blocker for blocker in blockers)
        if hard_memory_failure or runtime_failure:
            verdict = "NO_FIT"
        elif blockers:
            verdict = "CONSTRAINT_FAIL"
        elif performance_unknown or power_unknown:
            verdict = "BENCHMARK_REQUIRED"
        elif exact_performance:
            verdict = "VERIFIED_FIT"
        else:
            verdict = "FEASIBLE"

        quality = 0.5
        if model.accuracy is not None:
            quality = (model.accuracy.value - min_acc) / acc_span
        memory_headroom = max(
            0.0,
            min(1.0, (available_memory - estimated_memory) / max(available_memory, 0.1)),
        )
        performance_score = 0.35
        if exact_performance and benchmark and benchmark.latency_ms is not None:
            performance_score = min(1.0, 40.0 / max(benchmark.latency_ms, 1.0))
        runtime_score = 1.0 if runtime_ready else 0.45
        score = 40.0 * quality + 25.0 * memory_headroom + 25.0 * performance_score + 10.0 * runtime_score
        if verdict == "NO_FIT":
            score -= 80.0
        elif verdict == "CONSTRAINT_FAIL":
            score -= 45.0
        elif verdict == "BENCHMARK_REQUIRED":
            score -= 25.0

        recommendations.append(
            Recommendation(
                model=model,
                verdict=verdict,
                score=round(score, 2),
                runtime=runtime,
                precision=precision,
                estimated_memory_gb=estimated_memory,
                memory_evidence=memory_evidence,
                benchmark=benchmark,
                evidence_match=match,
                evidence_confidence=_confidence(match),  # type: ignore[arg-type]
                runtime_available=runtime_ready,
                reasons=tuple(reasons),
                blockers=tuple(blockers),
                registry=loaded.provenance,
            )
        )
    return sorted(recommendations, key=lambda item: item.score, reverse=True)
