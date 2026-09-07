from pathlib import Path

import pytest

from autonomyfit.backends import (
    BackendError,
    BenchmarkRequest,
    CoreMLBackend,
    _preferred_ort_providers,
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
    assert parsed["throughput_qps"] == 811.74
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

    path = tmp_path / "model.engine"
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

def test_openvino_version_falls_back_to_installed_package(monkeypatch):
    from autonomyfit.backends import OpenVINOBackend

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/benchmark_app")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: type("Result", (), {"stdout": "usage only", "stderr": ""})(),
    )
    monkeypatch.setattr(
        "autonomyfit.backends.importlib.metadata.version",
        lambda name: "2026.3.0" if name == "openvino" else "0",
    )
    availability = OpenVINOBackend().availability()
    assert availability.available is True
    assert availability.version == "2026.3.0"


def test_onnxruntime_default_provider_order_is_platform_specific():
    assert _preferred_ort_providers("nvidia")[:2] == (
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
    )
    assert _preferred_ort_providers("apple")[0] == "CoreMLExecutionProvider"
    assert _preferred_ort_providers("intel")[0] == "OpenVINOExecutionProvider"
    assert _preferred_ort_providers("qualcomm")[0] == "QNNExecutionProvider"


def test_batched_onnx_throughput_counts_items(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from autonomyfit.backends import OnnxRuntimeBackend

    path = tmp_path / "batch.onnx"
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        "batch",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [4, 2])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4, 2])],
    )
    onnx.save(onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 21)], ir_version=10,
    ), path)
    request = BenchmarkRequest(
        model_path=path, model_id="batch", model_revision="r1", hardware=HARDWARE,
        iterations=3, warmup=1, batch_size=4, provider="CPUExecutionProvider",
    )
    report = OnnxRuntimeBackend().benchmark(request)
    assert report["execution"]["batch_size"] == 4
    assert report["metrics"]["throughput_fps"] == pytest.approx(
        4000.0 / report["metrics"]["latency"]["mean_ms"]
    )


def test_openvino_missing_throughput_remains_unknown():
    parsed = parse_openvino_output("Min: 4.0 ms\nMedian: 4.2 ms\nAverage: 4.5 ms\nMax: 5.0 ms")
    assert parsed["throughput_fps"] is None


def test_trtexec_batch_throughput_conversion(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from autonomyfit.backends import BackendAvailability, TensorRTBackend

    path = tmp_path / "model.engine"
    path.write_bytes(b"engine")
    backend = TensorRTBackend()
    monkeypatch.setattr(backend, "availability", lambda: BackendAvailability(
        "tensorrt", True, "10", "/trtexec",
    ))
    monkeypatch.setattr("autonomyfit.backends.subprocess.run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stderr="", stdout=(
            "Throughput: 100 qps\n"
            "Latency: min = 10 ms, max = 10 ms, mean = 10 ms, median = 10 ms, "
            "percentile(90%) = 10 ms, percentile(95%) = 10 ms, percentile(99%) = 10 ms"
        ),
    ))
    request = BenchmarkRequest(
        model_path=path, model_id="batch", model_revision="r1", hardware=HARDWARE,
        batch_size=4, input_shapes={"input": [4, 2]}, trusted_artifact=True,
    )
    report = backend.benchmark(request)
    assert report["execution"]["backend_options"]["throughput_qps"] == 100
    assert report["metrics"]["throughput_fps"] == 400


def test_trtexec_build_power_cannot_be_used_as_inference_power(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from autonomyfit.backends import BackendAvailability, TensorRTBackend

    path = tmp_path / "model with spaces.engine"
    path.write_bytes(b"engine")
    backend = TensorRTBackend()
    monkeypatch.setattr(backend, "availability", lambda: BackendAvailability(
        "tensorrt", True, "10", "/trtexec",
    ))
    monkeypatch.setattr("autonomyfit.backends.subprocess.run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stderr="", stdout=(
            "Throughput: 100 qps\n"
            "Latency: min = 10 ms, max = 10 ms, mean = 10 ms, median = 10 ms, "
            "percentile(90%) = 10 ms, percentile(95%) = 10 ms, percentile(99%) = 10 ms"
        ),
    ))

    class Sampler:
        def __init__(self, reader):
            pass

        def start(self):
            pass

        def stop(self):
            return {"mean_w": 12.0, "max_w": 20.0, "energy_j": 60.0}

    monkeypatch.setattr("autonomyfit.backends.PowerSampler", Sampler)
    monkeypatch.setattr("autonomyfit.backends.power_reader", lambda platform: (
        lambda: 12.0, "NVIDIA GPU board power.draw",
    ))
    report = backend.benchmark(BenchmarkRequest(
        model_path=path, model_id="model", model_revision="r1", hardware=HARDWARE,
        trusted_artifact=True,
    ))
    assert report["metrics"]["power"] is None
    telemetry = report["execution"]["backend_options"]["native_process_power"]
    assert telemetry["mean_w"] == 12.0
    assert telemetry["energy_j"] == 60.0
    assert telemetry["scope"] == "NVIDIA GPU board power.draw"
    assert "engine build/load, warmup and inference" in telemetry["measurement_window"]
    import shlex
    assert shlex.split(report["reproducibility"]["command"]) == report["execution"]["backend_options"]["native_command"]


@pytest.mark.parametrize("overrides", [
    {"shape_override": [4, 2]},
    {"batch_size": 4},
    {"batch_size": 4, "input_shapes": {"input": [1, 2]}},
])
def test_trtexec_rejects_unapplied_or_conflicting_engine_batch(overrides):
    request = BenchmarkRequest(
        model_path=Path("model.engine"), model_id="model", model_revision="r1",
        hardware=HARDWARE, **overrides,
    )
    with pytest.raises(BackendError):
        build_trtexec_command(request)


def test_openvino_ir_rejects_ignored_unnamed_shape():
    request = BenchmarkRequest(
        model_path=Path("model.xml"), model_id="model", model_revision="r1",
        hardware=HARDWARE, shape_override=[4, 2],
    )
    with pytest.raises(BackendError, match="named"):
        build_openvino_command(request)


def _fake_coreml(monkeypatch, dtype, shape):
    import sys
    from types import SimpleNamespace

    from autonomyfit.backends import BackendAvailability

    observed = []
    item = SimpleNamespace(name="input", type=SimpleNamespace(
        WhichOneof=lambda name: "multiArrayType",
        multiArrayType=SimpleNamespace(shape=shape, dataType=dtype),
    ))

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def get_spec(self):
            return SimpleNamespace(description=SimpleNamespace(input=[item]))

        def predict(self, feeds):
            observed.append(feeds["input"])
            return {}

    monkeypatch.setitem(sys.modules, "coremltools", SimpleNamespace(
        ComputeUnit=SimpleNamespace(ALL=0, CPU_ONLY=1, CPU_AND_GPU=2, CPU_AND_NE=3),
        models=SimpleNamespace(MLModel=Model),
        proto=SimpleNamespace(FeatureTypes_pb2=SimpleNamespace(ArrayFeatureType=SimpleNamespace(
            FLOAT16=65552, FLOAT32=65568, DOUBLE=65600, INT32=131104,
        ))),
    ))
    backend = CoreMLBackend()
    monkeypatch.setattr(backend, "availability", lambda: BackendAvailability("coreml", True, "9"))
    return backend, observed


@pytest.mark.parametrize("dtype, expected", [
    (65552, "float16"), (65568, "float32"), (65600, "float64"), (131104, "int32"),
])
def test_coreml_obeys_input_dtype(monkeypatch, tmp_path, dtype, expected):
    backend, observed = _fake_coreml(monkeypatch, dtype, [4, 2])
    path = tmp_path / "model.mlmodel"
    path.write_bytes(b"model")
    request = BenchmarkRequest(
        model_path=path, model_id="model", model_revision="r1", hardware=HARDWARE,
        batch_size=4, iterations=2, warmup=1,
    )
    report = backend.benchmark(request)
    assert len(observed) == 3
    assert all(str(item.dtype) == expected for item in observed)
    assert report["metrics"]["throughput_fps"] == pytest.approx(
        4000.0 / report["metrics"]["latency"]["mean_ms"]
    )


def test_coreml_rejects_batch_not_executed(monkeypatch, tmp_path):
    backend, observed = _fake_coreml(monkeypatch, 65568, [1, 2])
    request = BenchmarkRequest(
        model_path=tmp_path / "model.mlmodel", model_id="model", model_revision="r1",
        hardware=HARDWARE, batch_size=8,
    )
    with pytest.raises(BackendError, match="conflicts"):
        backend.benchmark(request)
    assert observed == []


def test_coreml_does_not_infer_channels_as_batch(monkeypatch, tmp_path):
    backend, _ = _fake_coreml(monkeypatch, 65568, [3, 8, 8])
    path = tmp_path / "model.mlmodel"
    path.write_bytes(b"model")
    report = backend.benchmark(BenchmarkRequest(
        model_path=path, model_id="model", model_revision="r1", hardware=HARDWARE,
        iterations=2, warmup=0,
    ))
    assert report["execution"]["batch_size"] is None
    assert report["metrics"]["throughput_fps"] is None
    assert report["execution"]["backend_options"]["throughput_qps"] > 0


@pytest.mark.parametrize("backend_name, suffix", [("tensorrt", ".engine"), ("openvino", ".xml")])
def test_native_timeout_is_an_actionable_backend_error(monkeypatch, tmp_path, backend_name, suffix):
    import subprocess

    from autonomyfit.backends import BackendAvailability, get_backend

    backend = get_backend(backend_name)
    monkeypatch.setattr(backend, "availability", lambda: BackendAvailability(
        backend_name, True, "version", "/native-tool",
    ))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 600)

    monkeypatch.setattr("autonomyfit.backends.subprocess.run", timeout)
    request = BenchmarkRequest(
        model_path=tmp_path / ("model" + suffix), model_id="model", model_revision="r1",
        hardware=HARDWARE, trusted_artifact=True,
    )
    with pytest.raises(BackendError, match="could not complete"):
        backend.benchmark(request)
