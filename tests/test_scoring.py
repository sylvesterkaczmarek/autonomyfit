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


def test_exact_jetson_benchmark_can_verify_constraint():
    hardware = hardware_from_profile("jetson-orin-nx-16gb")
    items = recommend_models(
        hardware,
        Constraints(task="detection", min_fps=200, max_latency_ms=5),
    )
    yolo26n = next(item for item in items if item.model.id == "yolo26n")
    assert yolo26n.verdict == "VERIFIED_FIT"
    assert yolo26n.latency_ms == 4.13
    assert yolo26n.fps is not None and yolo26n.fps > 200


def test_unknown_performance_does_not_silently_pass():
    hardware = hardware_from_profile("jetson-orin-nx-16gb")
    items = recommend_models(hardware, Constraints(task="detection", min_fps=30))
    yolo26s = next(item for item in items if item.model.id == "yolo26s")
    assert yolo26s.verdict == "BENCHMARK_REQUIRED"
    assert yolo26s.benchmark is None


def test_measured_constraint_failure_is_reported():
    hardware = hardware_from_profile("jetson-orin-nano-super-8gb")
    items = recommend_models(hardware, Constraints(task="detection", min_fps=300))
    yolo26n = next(item for item in items if item.model.id == "yolo26n")
    assert yolo26n.verdict == "CONSTRAINT_FAIL"
    assert any("measured" in blocker for blocker in yolo26n.blockers)


def test_vlm_published_memory_can_fail_small_machine():
    hardware = _cpu_with_ram(0.75)
    items = recommend_models(hardware, Constraints(task="vlm"))
    smol = next(item for item in items if item.model.id == "smolvlm-256m-instruct")
    assert smol.verdict == "NO_FIT"
    assert smol.memory_evidence == "published"


def test_power_constraint_requires_measurement_when_missing():
    hardware = hardware_from_profile("jetson-orin-nx-16gb")
    items = recommend_models(hardware, Constraints(task="detection", max_power_w=15))
    yolo26n = next(item for item in items if item.model.id == "yolo26n")
    assert yolo26n.verdict == "BENCHMARK_REQUIRED"
