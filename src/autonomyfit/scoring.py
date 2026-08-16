from __future__ import annotations

from pathlib import Path

from .catalog import load_benchmarks, load_model_catalog
from .models import (
    BenchmarkRecord,
    Constraints,
    HardwareProfile,
    ModelProfile,
    Recommendation,
)
from .registry import RegistryClient


def _runtime_available(hardware: HardwareProfile, runtime: str) -> bool:
    aliases = {
        "onnx": "onnxruntime",
        "pytorch": "pytorch",
        "tensorrt": "tensorrt",
        "coreml": "coreml",
        "transformers": "transformers",
    }
    capability_name = aliases.get(runtime, runtime)
    return any(cap.name == capability_name and cap.available for cap in hardware.runtimes)


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
        return max(1.0, weight_gib * 16.0 + 0.75), "screening estimate"
    return max(1.5, weight_gib * 1.6 + 1.0), "screening estimate"


def _available_accelerator_memory(hardware: HardwareProfile) -> float:
    if hardware.accelerator_memory_gb is not None:
        return hardware.accelerator_memory_gb
    return hardware.ram_available_gb


def _benchmark_for(
    benchmarks: list[BenchmarkRecord],
    hardware: HardwareProfile,
    model_id: str,
    runtime: str,
    precision: str,
) -> BenchmarkRecord | None:
    if not hardware.matched_profile:
        return None
    candidates = [
        record
        for record in benchmarks
        if record.hardware_id == hardware.matched_profile
        and record.model_id == model_id
        and record.runtime == runtime
        and record.precision.lower() == precision.lower()
    ]
    return candidates[0] if candidates else None


def recommend_models(
    hardware: HardwareProfile,
    constraints: Constraints,
    catalog_path: Path | None = None,
    *,
    offline: bool = False,
    force_registry_refresh: bool = False,
    registry_client: RegistryClient | None = None,
) -> list[Recommendation]:
    loaded = load_model_catalog(
        catalog_path,
        offline=offline,
        force_refresh=force_registry_refresh,
        client=registry_client,
    )
    models = [model for model in loaded.models if model.task == constraints.task]
    benchmarks = load_benchmarks()
    recommendations: list[Recommendation] = []

    task_accuracy = [model.accuracy.value for model in models if model.accuracy is not None]
    min_acc = min(task_accuracy) if task_accuracy else 0.0
    max_acc = max(task_accuracy) if task_accuracy else 1.0
    acc_span = max(max_acc - min_acc, 1e-9)

    available_memory = _available_accelerator_memory(hardware)

    for model in models:
        runtime = choose_runtime(model, hardware, constraints.runtime)
        precision = choose_precision(model, hardware, constraints.precision)
        estimated_memory, memory_evidence = estimate_memory_gb(model, precision)
        benchmark = _benchmark_for(benchmarks, hardware, model.id, runtime, precision)
        runtime_ready = _runtime_available(hardware, runtime)

        reasons: list[str] = []
        blockers: list[str] = []
        performance_unknown = False
        power_unknown = False

        if runtime not in model.runtimes:
            blockers.append(f"{runtime} is not listed for this model profile")

        if estimated_memory > available_memory:
            blockers.append(
                f"memory screen needs {estimated_memory:.2f} GB; "
                f"{available_memory:.2f} GB is available"
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
                    f"{model.accuracy.name} {model.accuracy.value:.2f} is below "
                    f"{constraints.min_accuracy:.2f}"
                )
            else:
                reasons.append(
                    f"{model.accuracy.name} {model.accuracy.value:.2f} "
                    "meets the accuracy constraint"
                )

        if constraints.min_fps is not None or constraints.max_latency_ms is not None:
            if benchmark is None:
                performance_unknown = True
            else:
                if constraints.min_fps is not None and benchmark.fps < constraints.min_fps:
                    blockers.append(
                        f"measured {benchmark.fps:.1f} FPS is below {constraints.min_fps:.1f} FPS"
                    )
                if (
                    constraints.max_latency_ms is not None
                    and benchmark.latency_ms > constraints.max_latency_ms
                ):
                    blockers.append(
                        f"measured {benchmark.latency_ms:.2f} ms exceeds "
                        f"{constraints.max_latency_ms:.2f} ms"
                    )
                if not any("measured" in blocker for blocker in blockers):
                    reasons.append(
                        f"published matching benchmark is {benchmark.latency_ms:.2f} ms "
                        f"({benchmark.fps:.1f} FPS)"
                    )

        if constraints.max_power_w is not None:
            if benchmark and benchmark.power_w is not None:
                if benchmark.power_w > constraints.max_power_w:
                    blockers.append(
                        f"measured {benchmark.power_w:.1f} W exceeds "
                        f"{constraints.max_power_w:.1f} W"
                    )
                else:
                    reasons.append(f"measured power {benchmark.power_w:.1f} W meets the limit")
            else:
                power_unknown = True

        if runtime_ready:
            reasons.append(
                f"{runtime} is supported by the selected hardware profile"
                if hardware.os_name == "profile"
                else f"{runtime} is available on this environment"
            )
        else:
            reasons.append(f"{runtime} is a target runtime but is not currently installed")

        hard_memory_failure = any("memory screen needs" in blocker for blocker in blockers)
        if hard_memory_failure or runtime not in model.runtimes:
            verdict = "NO_FIT"
        elif blockers:
            verdict = "CONSTRAINT_FAIL"
        elif performance_unknown or power_unknown:
            verdict = "BENCHMARK_REQUIRED"
        elif benchmark is not None:
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
        if benchmark is not None:
            performance_score = min(1.0, 40.0 / max(benchmark.latency_ms, 1.0))
        runtime_score = 1.0 if runtime_ready else 0.45

        score = (
            40.0 * quality
            + 25.0 * memory_headroom
            + 25.0 * performance_score
            + 10.0 * runtime_score
        )
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
                runtime_available=runtime_ready,
                reasons=tuple(reasons),
                blockers=tuple(blockers),
                registry=loaded.provenance,
            )
        )

    return sorted(recommendations, key=lambda item: item.score, reverse=True)
