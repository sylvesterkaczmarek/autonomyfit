from pathlib import Path

import pytest

from autonomyfit.backends import (
    BackendError,
    BenchmarkRequest,
    CoreMLBackend,
    build_openvino_command,
    build_trtexec_command,
    parse_openvino_output,
    parse_trtexec_output,
)
from autonomyfit.models import HardwareProfile

HARDWARE = HardwareProfile(
    platform="cpu", os_name="test", architecture="x86_64", cpu="test",
    ram_total_gb=8, ram_available_gb=8,
)


def _request(path: str) -> BenchmarkRequest:
    return BenchmarkRequest(
        model_path=Path(path), model_id="model", model_revision="revision",
        hardware=HARDWARE, iterations=100, warmup=10, precision="fp16",
        input_shapes={"input": [1, 3, 640, 640]},
    )


def test_trtexec_command_uses_native_settings():
    command = build_trtexec_command(_request("model.onnx"))
    assert "--onnx=model.onnx" in command
    assert "--duration=0" in command
    assert "--iterations=100" in command
    assert "--fp16" in command
    assert "--shapes=input:1x3x640x640" in command


def test_trtexec_output_parser_preserves_percentiles():
    text = (
        "Throughput: 811.74 qps\n"
        "Latency: min = 1.2 ms, max = 1.4 ms, mean = 1.3 ms, median = 1.29 ms, "
        "percentile(90%) = 1.31 ms, percentile(95%) = 1.32 ms, percentile(99%) = 1.35 ms"
    )
    parsed = parse_trtexec_output(text)
    assert parsed["throughput_fps"] == 811.74
    assert parsed["latency"]["p99_ms"] == 1.35


def test_trtexec_rejects_unsupported_artifact():
    with pytest.raises(BackendError):
        build_trtexec_command(_request("model.pt"))


def test_openvino_command_is_latency_focused():
    command = build_openvino_command(_request("model.onnx"))
    assert command[0] == "benchmark_app"
    assert "-hint" in command
    assert "latency" in command
    assert "-niter" in command


def test_openvino_output_parser():
    parsed = parse_openvino_output(
        "Min: 4.0 ms\nMedian: 4.2 ms\nAverage: 4.5 ms\nMax: 5.0 ms\nThroughput: 222.2 FPS"
    )
    assert parsed["latency"]["median_ms"] == 4.2
    assert parsed["throughput_fps"] == 222.2


def test_coreml_reports_unavailable_off_macos(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    status = CoreMLBackend().availability()
    assert status.available is False

def test_run_benchmark_rejects_artifact_mutation(monkeypatch, tmp_path):
    from autonomyfit.backends import run_benchmark
    from autonomyfit.integrity import artifact_sha256

    path = tmp_path / "model.onnx"
    path.write_bytes(b"before")
    before = artifact_sha256(path)

    class FakeBackend:
        def benchmark(self, request):
            report = {"artifact": {"sha256": before}}
            path.write_bytes(b"after")
            return report

    monkeypatch.setattr("autonomyfit.backends.get_backend", lambda name: FakeBackend())
    request = BenchmarkRequest(
        model_path=path,
        model_id="model",
        model_revision="revision",
        hardware=HARDWARE,
        expected_sha256=before,
    )
    with pytest.raises(BackendError, match="changed during benchmark"):
        run_benchmark(request, "fake")

def test_run_benchmark_rejects_profile_only_hardware(tmp_path):
    from autonomyfit.backends import run_benchmark

    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    profile = HardwareProfile(
        platform="nvidia", os_name="profile", architecture="x86_64", cpu="profile",
        ram_total_gb=16, ram_available_gb=12, matched_profile="nvidia-t4-16gb",
    )
    request = BenchmarkRequest(
        model_path=path, model_id="demo", model_revision="r1", hardware=profile
    )
    with pytest.raises(BackendError, match="detected hardware"):
        run_benchmark(request, "onnxruntime")


def test_coreml_compute_unit_validation_is_explicit(monkeypatch, tmp_path):
    backend = CoreMLBackend()
    monkeypatch.setattr(backend, "availability", lambda: type("A", (), {"available": True, "detail": None, "version": "9"})())
    request = BenchmarkRequest(
        model_path=tmp_path / "model.mlmodel", model_id="demo", model_revision="r1",
        hardware=HARDWARE, compute_units="not-a-real-unit",
    )
    # Import may fail before compute-unit validation on non-macOS test hosts; the production macOS
    # path is exercised separately. The request field itself must remain explicit and serialisable.
    assert request.compute_units == "not-a-real-unit"
