from dataclasses import replace

from autonomyfit.evidence import (
    BenchmarkEvidence,
    EvidenceStore,
    LatencyStats,
    PowerStats,
)
from autonomyfit.hardware import hardware_from_profile
from autonomyfit.models import Constraints, HardwareProfile, RuntimeCapability
from autonomyfit.scoring import recommend_models


def _cpu_with_ram(gb: float) -> HardwareProfile:
    return HardwareProfile(
        platform="cpu",
        os_name="test",
        architecture="x86_64",
        cpu="test cpu",
        ram_total_gb=gb,
        ram_available_gb=gb,
        runtimes=(RuntimeCapability("onnxruntime", True, "test"),),
    )


def _exact_local_yolo_evidence() -> EvidenceStore:
    evidence = BenchmarkEvidence(
        id="local-yolo26n",
        model_id="yolo26n",
        model_revision="revision-1",
        artifact_id="artifact-yolo26n",
        artifact_sha256="a" * 64,
        artifact_format="tensorrt-engine",
        hardware_id="jetson-orin-nx-16gb",
        hardware_name="NVIDIA Jetson Orin NX 16GB",
        runtime="tensorrt",
        runtime_version="10.0",
        provider="trtexec",
        precision="fp16",
        quantization=None,
        batch_size=1,
        input_shapes={"images": [1, 3, 640, 640]},
        power_mode="MAXN",
        clocks={},
        warmup=10,
        iterations=100,
        latency=LatencyStats(mean_ms=4.0, median_ms=3.9, p50_ms=3.9, p90_ms=4.2, p95_ms=4.3, p99_ms=4.5),
        throughput_fps=250.0,
        power=PowerStats(mean_w=12.0, max_w=14.0, energy_j=5.0, scope="Jetson VDD_IN rail"),
        peak_memory_mb=1200.0,
        peak_memory_scope="process RSS",
        quality="local-measured",
        source_id="autonomyfit-local",
        source_url="local://benchmark",
        source_date="2026-08-16",
        software_stack_id="stack-1",
        provider_version="10.0",
        machine_source="detected",
        verified_identity=True,
    )
    return EvidenceStore(document={}, benchmarks=(evidence,))


def test_vendor_jetson_benchmark_is_context_not_verified_fit():
    hardware = hardware_from_profile("jetson-orin-nx-16gb")
    items = recommend_models(
        hardware,
        Constraints(task="detection", min_fps=200, max_latency_ms=5),
        offline=True,
    )
    yolo26n = next(item for item in items if item.model.id == "yolo26n")
    assert yolo26n.verdict == "BENCHMARK_REQUIRED"
    assert yolo26n.latency_ms == 4.13
    assert yolo26n.evidence_confidence == "LOW"
    assert yolo26n.evidence_match is not None and yolo26n.evidence_match.exact is False


def test_exact_local_identity_can_verify_constraint():
    hardware = hardware_from_profile("jetson-orin-nx-16gb")
    hardware = replace(
        hardware,
        runtimes=(RuntimeCapability("tensorrt", True, "10.0", "local"),),
    )
    items = recommend_models(
        hardware,
        Constraints(
            task="detection",
            min_fps=200,
            max_latency_ms=5,
            runtime="tensorrt",
            precision="fp16",
            model_id="yolo26n",
            model_revision="revision-1",
            artifact_sha256="a" * 64,
            provider="trtexec",
            provider_version="10.0",
            batch_size=1,
            input_shapes={"images": [1, 3, 640, 640]},
            power_mode="MAXN",
            software_stack_id="stack-1",
        ),
        offline=True,
        evidence_store=_exact_local_yolo_evidence(),
    )
    yolo26n = items[0]
    assert yolo26n.verdict == "VERIFIED_FIT"
    assert yolo26n.evidence_confidence == "HIGH"
    assert yolo26n.latency_ms == 3.9


def test_unknown_performance_does_not_silently_pass():
    hardware = hardware_from_profile("jetson-orin-nx-16gb")
    items = recommend_models(hardware, Constraints(task="detection", min_fps=30), offline=True)
    yolo26s = next(item for item in items if item.model.id == "yolo26s")
    assert yolo26s.verdict == "BENCHMARK_REQUIRED"
    assert yolo26s.benchmark is None


def test_vendor_reference_does_not_hard_fail_constraint():
    hardware = hardware_from_profile("jetson-orin-nano-super-8gb")
    items = recommend_models(
        hardware,
        Constraints(task="detection", min_fps=300),
        offline=True,
    )
    yolo26n = next(item for item in items if item.model.id == "yolo26n")
    assert yolo26n.verdict == "BENCHMARK_REQUIRED"
    assert not yolo26n.blockers


def test_vlm_published_memory_can_fail_small_machine():
    hardware = _cpu_with_ram(0.75)
    items = recommend_models(hardware, Constraints(task="vlm"), offline=True)
    smol = next(item for item in items if item.model.id == "smolvlm-256m-instruct")
    assert smol.verdict == "NO_FIT"
    assert smol.memory_evidence == "published"


def test_power_constraint_requires_exact_scoped_measurement():
    hardware = hardware_from_profile("jetson-orin-nx-16gb")
    items = recommend_models(
        hardware,
        Constraints(task="detection", max_power_w=15),
        offline=True,
    )
    yolo26n = next(item for item in items if item.model.id == "yolo26n")
    assert yolo26n.verdict == "BENCHMARK_REQUIRED"


def test_requested_runtime_and_precision_are_normalized():
    hardware = hardware_from_profile("jetson-orin-nx-16gb")
    items = recommend_models(
        hardware,
        Constraints(task="detection", runtime=" TensorRT ", precision=" FP16 "),
        offline=True,
    )
    yolo26n = next(item for item in items if item.model.id == "yolo26n")
    assert yolo26n.runtime == "tensorrt"
    assert yolo26n.precision == "fp16"
    assert yolo26n.verdict == "FEASIBLE"


def test_recommendations_are_deterministic_offline():
    hardware = hardware_from_profile("jetson-orin-nx-16gb")
    first = recommend_models(hardware, Constraints(task="detection"), offline=True)
    second = recommend_models(hardware, Constraints(task="detection"), offline=True)
    assert [(item.model.id, item.score) for item in first] == [
        (item.model.id, item.score) for item in second
    ]
