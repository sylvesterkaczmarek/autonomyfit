from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from autonomyfit.artifacts import ManagedArtifact
from autonomyfit.backends import BackendError
from autonomyfit.deployment import (
    DeploymentValidationError,
    ValidationOptions,
    structural_checks,
    validate_deployment,
)
from autonomyfit.evidence import (
    BenchmarkEvidence,
    EvidenceStore,
    LatencyStats,
    PowerStats,
)
from autonomyfit.hardware import hardware_from_profile
from autonomyfit.models import (
    Constraints,
    HardwareProfile,
    ModelProfile,
    RuntimeCapability,
)
from autonomyfit.scoring import recommend_models


def _model() -> ModelProfile:
    return ModelProfile(
        id="demo",
        display_name="Demo",
        family="demo",
        task="detection",
        params_m=1,
        source_id="demo",
        source_url="https://example.com/demo",
        runtimes=("tensorrt",),
        supported_precisions=("fp16",),
        license_spdx="Apache-2.0",
        license_status="published",
    )


def test_untrusted_serialized_tensorrt_engine_fails_trust_boundary(tmp_path):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    artifact = ManagedArtifact(
        model_id="demo",
        path=engine,
        format="tensorrt-engine",
        sha256="a" * 64,
        size_bytes=6,
        source="local",
        provenance_url=None,
        requested_revision="r1",
        resolved_revision="r1",
        filename=engine.name,
        license_spdx="Apache-2.0",
        license_status="published",
        remote_code_required=False,
        trusted_for_execution=False,
        cached=False,
        acquired_at="2026-08-16T12:00:00Z",
    )
    checks, _ = structural_checks(
        artifact, runtime="tensorrt", runtime_available=True, model=_model()
    )
    trust = next(item for item in checks if item["name"] == "tensorrt-trust-boundary")
    assert trust["status"] == "fail"


def test_validate_without_artifact_returns_safe_selection_report():
    report = validate_deployment(
        ValidationOptions(
            model_id="yolo26n",
            offline=True,
            hardware_profile="nvidia-t4-16gb",
        )
    )
    assert report["status"] == "artifact-selection-required"
    assert report["artifact"] is None
    assert report["model"]["license_status"]
    assert report["reproducibility"]["commands"] == ["autonomyfit scan"]


def _fake_hardware() -> HardwareProfile:
    return HardwareProfile(
        platform="cpu",
        os_name="Linux",
        architecture="x86_64",
        cpu="CI CPU",
        ram_total_gb=16,
        ram_available_gb=12,
        runtimes=(RuntimeCapability("onnxruntime", True, "1.20.0", provider="CPUExecutionProvider"),),
        supported_precisions=("fp32",),
    )


def _fake_benchmark(path: Path):
    return {
        "schema_version": 2,
        "benchmark_id": "local-1234567890",
        "created_at": "2026-08-16T13:00:00Z",
        "quality": "local-measured",
        "notes": None,
        "model": {"id": "yolo26n", "revision": None},
        "artifact": {"path": path.name, "format": "onnx", "sha256": "a" * 64, "size_bytes": 4},
        "hardware": {"id": "local-cpu-demo", "platform": "cpu", "device": None, "cpu": "CI CPU", "ram_total_gb": 16, "os": "Linux", "architecture": "x86_64", "driver": None, "power_mode": None, "clocks": {}, "thermal_c": {}},
        "software": {"runtime": "onnxruntime", "runtime_version": "1.20.0", "provider": "CPUExecutionProvider", "provider_version": "1.20.0", "python_version": "3.12", "autonomyfit_version": "0.6.0"},
        "execution": {"precision": "fp32", "quantization": None, "batch_size": 1, "input_shapes": {"images": [1, 3, 640, 640]}, "warmup": 1, "iterations": 3, "random_seed": 0, "backend_options": {}},
        "metrics": {"load_ms": 1.0, "latency": {"min_ms": 4.0, "mean_ms": 5.0, "median_ms": 5.0, "p50_ms": 5.0, "p90_ms": 5.5, "p95_ms": 5.6, "p99_ms": 5.7, "max_ms": 6.0, "stdev_ms": 0.5}, "throughput_fps": 200.0, "peak_memory_mb": 100.0, "peak_memory_scope": "process RSS", "power": None},
        "reproducibility": {"command": "autonomyfit validate yolo26n --benchmark", "hostname_hash": "abc", "environment_fingerprint": "deadbeef"},
    }


def test_benchmark_success_and_failure_are_reflected_in_validation(monkeypatch, tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx")
    hardware = _fake_hardware()
    monkeypatch.setattr("autonomyfit.deployment._resolve_hardware", lambda profile: hardware)
    monkeypatch.setattr("autonomyfit.deployment._check_onnx", lambda path: ("pass", "synthetic test model"))
    monkeypatch.setattr("autonomyfit.deployment.run_benchmark", lambda request, backend: _fake_benchmark(path))
    report = validate_deployment(
        ValidationOptions(
            model_id="yolo26n",
            artifact=path,
            runtime="onnx",
            precision="fp32",
            benchmark=True,
            import_local=False,
            offline=True,
            iterations=3,
            warmup=1,
        )
    )
    assert report["status"] == "validated"
    assert report["benchmark"]["metrics"]["latency"]["median_ms"] == 5.0
    assert any(item["name"] == "benchmark" and item["status"] == "pass" for item in report["compatibility"]["checks"])

    def fail(request, backend):
        raise BackendError("runtime failed")

    monkeypatch.setattr("autonomyfit.deployment.run_benchmark", fail)
    failed = validate_deployment(
        ValidationOptions(
            model_id="yolo26n",
            artifact=path,
            runtime="onnx",
            precision="fp32",
            benchmark=True,
            import_local=False,
            offline=True,
        )
    )
    assert failed["status"] == "failed"
    assert any("runtime failed" in item["detail"] for item in failed["compatibility"]["checks"])


def test_exact_local_evidence_precedes_generic_vendor_reference():
    hardware = replace(
        hardware_from_profile("jetson-orin-nx-16gb"),
        runtimes=(RuntimeCapability("tensorrt", True, "10.0", "local"),),
    )
    local = BenchmarkEvidence(
        id="local", model_id="yolo26n", model_revision="r1", artifact_id="a",
        artifact_sha256="a" * 64, artifact_format="onnx", hardware_id="jetson-orin-nx-16gb",
        hardware_name="Jetson", runtime="tensorrt", runtime_version="10.0", provider="trtexec",
        precision="fp16", quantization=None, batch_size=1,
        input_shapes={"input": [1, 3, 640, 640]}, power_mode=None,
        clocks={}, warmup=5, iterations=10, latency=LatencyStats(mean_ms=3.0, median_ms=3.0),
        throughput_fps=333.3, power=PowerStats(), peak_memory_mb=None, peak_memory_scope=None,
        quality="local-measured", source_id="local", source_url="local://benchmark",
        source_date="2026-08-16", software_stack_id="stack-1", provider_version="trtexec-1",
        machine_source="detected", verified_identity=True,
    )
    vendor = replace(
        local,
        id="vendor",
        quality="vendor-published",
        source_id="vendor",
        source_url="https://example.com/vendor",
        latency=LatencyStats(mean_ms=10.0, median_ms=10.0),
        throughput_fps=100.0,
    )
    items = recommend_models(
        hardware,
        Constraints(
            task="detection", model_id="yolo26n", model_revision="r1",
            artifact_sha256="a" * 64, runtime="tensorrt", precision="fp16",
            provider="trtexec", provider_version="trtexec-1", batch_size=1,
            input_shapes={"input": [1, 3, 640, 640]}, software_stack_id="stack-1",
            max_latency_ms=5.0,
        ),
        offline=True,
        evidence_store=EvidenceStore(document={}, benchmarks=(vendor, local)),
    )
    assert items[0].benchmark is not None
    assert items[0].benchmark.id == "local"
    assert items[0].verdict == "VERIFIED_FIT"

def test_candidate_assessment_reranks_exact_supplied_artifacts(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from autonomyfit.deployment import assess_candidates

    first = replace(
        _model(), id="first", source_url="https://example.com/first",
        runtimes=("onnx",), supported_precisions=("fp32",),
    )
    second = replace(
        _model(), id="second", source_url="https://example.com/second",
        runtimes=("onnx",), supported_precisions=("fp32",),
    )
    hashes = {"first": "1" * 64, "second": "2" * 64}

    def fake_validate(options):
        return {
            "model": {"id": options.model_id, "revision": f"rev-{options.model_id}"},
            "artifact": {"sha256": hashes[options.model_id]},
        }

    calls = []

    def fake_recommend(hardware, constraints, **kwargs):
        calls.append(
            (constraints.model_id, constraints.model_revision, constraints.artifact_sha256)
        )
        return [SimpleNamespace(model=SimpleNamespace(id=constraints.model_id))]

    monkeypatch.setattr("autonomyfit.deployment.validate_deployment", fake_validate)
    monkeypatch.setattr("autonomyfit.deployment._resolve_hardware", lambda profile: _fake_hardware())
    monkeypatch.setattr(
        "autonomyfit.deployment.load_model_catalog",
        lambda **kwargs: SimpleNamespace(models=(first, second)),
    )
    monkeypatch.setattr("autonomyfit.deployment.recommend_models", fake_recommend)
    monkeypatch.setattr("autonomyfit.deployment.rank_recommendations", lambda items, objective: items)
    monkeypatch.setattr(
        "autonomyfit.deployment.recommendation_dict",
        lambda item: {"model_id": item.model.id},
    )

    result = assess_candidates(
        ["first", "second"],
        {"first": tmp_path / "first.onnx", "second": tmp_path / "second.onnx"},
        runtime="onnx",
        precision="fp32",
        offline=True,
    )
    assert calls == [
        ("first", "rev-first", hashes["first"]),
        ("second", "rev-second", hashes["second"]),
    ]
    assert [item["model_id"] for item in result["reordered_recommendations"]] == [
        "first", "second"
    ]

def test_profile_benchmark_mismatch_is_refused(monkeypatch):
    actual = _fake_hardware()
    monkeypatch.setattr("autonomyfit.deployment.detect_hardware", lambda: actual)
    try:
        validate_deployment(
            ValidationOptions(
                model_id="yolo26n", offline=True, benchmark=True,
                hardware_profile="nvidia-t4-16gb",
            )
        )
    except DeploymentValidationError as exc:
        assert "actual machine" in str(exc) or "does not match" in str(exc)
    else:
        raise AssertionError("profile-only benchmark should have been refused")
