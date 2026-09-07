from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autonomyfit.artifacts import ManagedArtifact
from autonomyfit.backends import BackendError
from autonomyfit.benchmark import hardware_evidence_id
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
from autonomyfit.integrity import artifact_sha256
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality": "local-measured",
        "notes": None,
        "model": {"id": "yolo26n", "revision": "1" * 40},
        "artifact": {"path": path.name, "format": "onnx", "sha256": artifact_sha256(path), "size_bytes": path.stat().st_size},
        "hardware": {"id": hardware_evidence_id(_fake_hardware()), "platform": "cpu", "device": None, "cpu": "CI CPU", "ram_total_gb": 16, "os": "Linux", "architecture": "x86_64", "driver": None, "power_mode": None, "clocks": {}, "thermal_c": {}},
        "software": {"runtime": "onnxruntime", "runtime_version": "1.20.0", "provider": "CPUExecutionProvider", "provider_version": "1.20.0", "python_version": "3.12", "autonomyfit_version": "0.6.0"},
        "execution": {"precision": "fp32", "quantization": None, "batch_size": 1, "input_shapes": {"images": [1, 3, 640, 640]}, "warmup": 1, "iterations": 3, "random_seed": 0, "backend_options": {}},
        "metrics": {"load_ms": 1.0, "latency": {"min_ms": 4.0, "mean_ms": 5.0, "median_ms": 5.0, "p50_ms": 5.0, "p90_ms": 5.5, "p95_ms": 5.6, "p99_ms": 5.7, "max_ms": 6.0, "stdev_ms": 0.5}, "throughput_fps": 200.0, "peak_memory_mb": 100.0, "peak_memory_scope": "process RSS", "power": None},
        "reproducibility": {"command": "autonomyfit validate yolo26n --benchmark", "hostname_hash": "abc", "environment_fingerprint": "deadbeef", "software_stack_fingerprint": "c" * 64},
        "measurement": {"machine_source": "detected", "profile_only": False, "artifact_identity_verified": True},
    }


def test_benchmark_success_and_failure_are_reflected_in_validation(monkeypatch, tmp_path, write_onnx):
    path = write_onnx(tmp_path / "model.onnx")
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


@pytest.mark.parametrize(
    "constraints,expected",
    [
        ({"max_latency_ms": 10.0}, "validated"),
        ({"max_latency_ms": 2.0}, "constraint-fail"),
        ({"max_power_w": 15.0}, "benchmark-required"),
    ],
)
@pytest.mark.parametrize("persist", [False, True])
def test_current_measurement_drives_constraints_with_or_without_persistence(
    monkeypatch, tmp_path, write_onnx, constraints, expected, persist
):
    path = write_onnx(tmp_path / "model.onnx")
    monkeypatch.setattr("autonomyfit.deployment._resolve_hardware", lambda profile: _fake_hardware())
    monkeypatch.setattr("autonomyfit.deployment.run_benchmark", lambda request, backend: _fake_benchmark(path))
    imported = []

    def save(report_path):
        imported.append(report_path)
        return tmp_path / "saved-evidence.json"

    monkeypatch.setattr("autonomyfit.deployment.import_benchmark_report", save)
    monkeypatch.setattr(
        "autonomyfit.scoring.load_evidence_store",
        lambda **kwargs: pytest.fail("current assessment must not depend on cached measurements"),
    )
    report = validate_deployment(ValidationOptions(
        model_id="yolo26n", artifact=path, runtime="onnx", precision="fp32",
        revision="1" * 40, benchmark=True, import_local=persist, offline=True, **constraints,
    ))
    assert report["status"] == expected
    assert len(imported) == int(persist)
    assert report["recommendation"]["latency_ms"] == 5.0


def test_unresolved_revision_cannot_validate_requested_latency(monkeypatch, tmp_path, write_onnx):
    path = write_onnx(tmp_path / "model.onnx")
    measured = _fake_benchmark(path)
    measured["model"]["revision"] = None
    monkeypatch.setattr("autonomyfit.deployment._resolve_hardware", lambda profile: _fake_hardware())
    monkeypatch.setattr("autonomyfit.deployment.run_benchmark", lambda request, backend: measured)
    report = validate_deployment(ValidationOptions(
        model_id="yolo26n", artifact=path, runtime="onnx", precision="fp32",
        benchmark=True, import_local=False, offline=True, max_latency_ms=10,
    ))
    assert report["status"] == "benchmark-required"


@pytest.mark.parametrize("settings", [
    {"max_latency_ms": float("nan")}, {"max_power_w": float("inf")},
    {"min_fps": -1}, {"iterations": 0}, {"warmup": -1}, {"shape": [0, 3]},
    {"offline": True, "artifact_url": "https://example.com/model.onnx"},
    {"artifact": Path("model.onnx"), "fetch": True},
])
def test_invalid_validation_settings_fail_before_work(settings):
    with pytest.raises(ValueError):
        ValidationOptions(model_id="yolo26n", **settings)


def test_safetensors_header_check_does_not_read_entire_weights(monkeypatch, tmp_path):
    from autonomyfit.deployment import _check_safetensors

    path = tmp_path / "large.safetensors"
    header = b'{"__metadata__":{}}'
    path.write_bytes(len(header).to_bytes(8, "little") + header)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("must read only the header"))
    assert _check_safetensors(path)[0] == "pass"


def test_failed_cache_write_preserves_measured_assessment(monkeypatch, tmp_path, write_onnx):
    path = write_onnx(tmp_path / "model.onnx")
    monkeypatch.setattr("autonomyfit.deployment._resolve_hardware", lambda profile: _fake_hardware())
    monkeypatch.setattr("autonomyfit.deployment.run_benchmark", lambda request, backend: _fake_benchmark(path))

    def fail(path):
        raise PermissionError("evidence directory is read-only")

    monkeypatch.setattr("autonomyfit.deployment.import_benchmark_report", fail)
    report = validate_deployment(ValidationOptions(
        model_id="yolo26n", artifact=path, runtime="onnx", precision="fp32",
        revision="1" * 40, benchmark=True, offline=True, max_latency_ms=10,
    ))
    assert report["status"] == "validated"
    assert report["recommendation"]["latency_ms"] == 5.0
    assert any("could not be saved" in warning for warning in report["warnings"])
    assert "local_evidence_path" not in report["benchmark"]


def test_failed_conversion_equivalence_prevents_benchmark_and_import(monkeypatch, tmp_path, write_onnx):
    from autonomyfit.conversions import ConversionResult

    path = write_onnx(tmp_path / "model.onnx")
    target = tmp_path / "converted.xml"
    target.write_text("<net/>")
    target.with_suffix(".bin").write_bytes(b"weights")
    converted = ConversionResult(
        source_path=path, source_sha256=artifact_sha256(path), source_format="onnx",
        target_path=target, target_sha256=artifact_sha256(target), target_format="openvino-ir",
        target_runtime="openvino", tool="test-converter", tool_version="1", command=(), duration_s=0.1,
    )
    monkeypatch.setattr("autonomyfit.deployment._resolve_hardware", lambda profile: _fake_hardware())
    monkeypatch.setattr("autonomyfit.deployment._runtime_capability", lambda *args: (True, "1", "test runtime"))
    monkeypatch.setattr("autonomyfit.deployment.convert_artifact", lambda *args, **kwargs: converted)
    monkeypatch.setattr("autonomyfit.deployment.compare_onnx_openvino_outputs", lambda *args, **kwargs: {"status": "failed", "reason": "non-finite outputs"})
    monkeypatch.setattr("autonomyfit.deployment.run_benchmark", lambda *args: pytest.fail("invalid converted output must not be benchmarked"))
    monkeypatch.setattr("autonomyfit.deployment.import_benchmark_report", lambda *args: pytest.fail("invalid conversion must not create evidence"))
    report = validate_deployment(ValidationOptions(
        model_id="yolo26n", artifact=path, runtime="openvino", precision="fp32",
        convert=True, benchmark=True, offline=True,
    ))
    assert report["status"] == "failed"
    assert report["benchmark"] is None
    assert any(check["name"] == "conversion-equivalence" and check["status"] == "fail" for check in report["compatibility"]["checks"])


def test_benchmark_reproduction_preserves_execution_settings(monkeypatch, tmp_path, write_onnx):
    import shlex

    path = write_onnx(tmp_path / "model with spaces.onnx")
    commands = []

    def measured(request, backend):
        commands.append(shlex.split(request.command))
        return _fake_benchmark(path)

    monkeypatch.setattr("autonomyfit.deployment._resolve_hardware", lambda profile: _fake_hardware())
    monkeypatch.setattr("autonomyfit.deployment.run_benchmark", measured)
    validate_deployment(ValidationOptions(
        model_id="yolo26n", artifact=path, runtime="onnx", precision="fp32", revision="1" * 40,
        benchmark=True, import_local=False, offline=True, iterations=7, warmup=2,
        provider="CPUExecutionProvider", device="CPU", shape=[1], max_latency_ms=10,
    ))
    command = commands[0]
    for flag, value in {
        "--artifact": str(path), "--sha256": artifact_sha256(path), "--revision": "1" * 40,
        "--iterations": "7", "--warmup": "2", "--provider": "CPUExecutionProvider",
        "--device": "CPU", "--shape": "1", "--latency-ms": "10",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--offline" in command and "--no-import-local" in command

def test_candidate_assessment_reranks_exact_supplied_artifacts(monkeypatch, tmp_path, write_onnx):
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
    artifacts = {name: tmp_path / f"{name}.onnx" for name in ("first", "second")}
    for index, path in enumerate(artifacts.values()):
        write_onnx(path, value=float(index))
    hashes = {name: artifact_sha256(path) for name, path in artifacts.items()}

    def fake_validate(options):
        benchmark = _fake_benchmark(options.artifact)
        benchmark["model"] = {"id": options.model_id, "revision": f"rev-{options.model_id}"}
        benchmark["benchmark_id"] = f"fresh-{options.model_id}"
        benchmark["local_evidence_path"] = str(tmp_path / f"{options.model_id}.json")
        return {
            "status": "validated",
            "model": {"id": options.model_id, "revision": f"rev-{options.model_id}"},
            "artifact": {"sha256": hashes[options.model_id]},
            "benchmark": benchmark,
        }

    calls = []

    def fake_recommend(hardware, constraints, **kwargs):
        store = kwargs["evidence_store"]
        assert len(store.benchmarks) == 1
        fresh = store.benchmarks[0]
        assert fresh.id == f"fresh-{constraints.model_id}"
        assert fresh.artifact_sha256 == constraints.artifact_sha256
        assert constraints.runtime == "onnxruntime"
        assert constraints.precision == "fp32"
        calls.append(
            (constraints.model_id, constraints.model_revision, constraints.artifact_sha256)
        )
        return [SimpleNamespace(model=SimpleNamespace(id=constraints.model_id), benchmark=fresh)]

    monkeypatch.setattr("autonomyfit.deployment.validate_deployment", fake_validate)
    monkeypatch.setattr("autonomyfit.deployment.detect_hardware", _fake_hardware)
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
        ["FIRST", "second"],
        artifacts,
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


@pytest.mark.parametrize("problem", [
    "different-tasks", "duplicate", "alias-duplicate", "unknown",
    "missing-mapping", "missing-file", "extra-mapping", "duplicate-mapping",
])
def test_candidate_input_errors_precede_any_execution(monkeypatch, tmp_path, problem):
    from types import SimpleNamespace

    from autonomyfit.deployment import assess_candidates

    first = replace(_model(), id="first", display_name="First model")
    second = replace(_model(), id="second", display_name="Second model")
    third = replace(_model(), id="third", display_name="Third model")
    names = ["first", "second"]
    artifacts = {name: tmp_path / f"{name}.onnx" for name in names}
    for path in artifacts.values():
        path.write_bytes(b"artifact")
    if problem == "different-tasks":
        second = replace(second, task="classification")
    elif problem == "duplicate":
        names = ["first", "FIRST"]
    elif problem == "alias-duplicate":
        names = ["first", "First model"]
    elif problem == "unknown":
        names = ["first", "unknown"]
    elif problem == "missing-mapping":
        del artifacts["second"]
    elif problem == "missing-file":
        artifacts["second"].unlink()
    elif problem == "extra-mapping":
        artifacts["third"] = artifacts["first"]
    elif problem == "duplicate-mapping":
        artifacts["First model"] = artifacts["first"]
    monkeypatch.setattr(
        "autonomyfit.deployment.load_model_catalog",
        lambda **kwargs: SimpleNamespace(models=(first, second, third)),
    )
    monkeypatch.setattr(
        "autonomyfit.deployment.validate_deployment",
        lambda options: pytest.fail("preflight must finish before candidate execution"),
    )
    with pytest.raises(DeploymentValidationError):
        assess_candidates(names, artifacts, offline=True)


@pytest.mark.parametrize("problem", ["failed", "missing", "invalid", "wrong-artifact", "no-match"])
def test_candidate_failure_cannot_fall_back_to_cached_or_unmeasured_rank(
    monkeypatch, tmp_path, problem, write_onnx
):
    from types import SimpleNamespace

    from autonomyfit.deployment import assess_candidates

    models = tuple(replace(_model(), id=name) for name in ("first", "second"))
    artifacts = {model.id: tmp_path / f"{model.id}.onnx" for model in models}
    for index, path in enumerate(artifacts.values()):
        write_onnx(path, value=float(index))

    def fake_validate(options):
        benchmark = _fake_benchmark(options.artifact)
        benchmark["model"] = {"id": options.model_id, "revision": "r1"}
        benchmark["benchmark_id"] = f"fresh-{options.model_id}"
        report = {
            "status": "validated", "model": benchmark["model"],
            "artifact": benchmark["artifact"], "benchmark": benchmark,
        }
        if options.model_id == "second":
            if problem == "failed":
                report["status"] = "failed"
            elif problem == "missing":
                report["benchmark"] = None
            elif problem == "invalid":
                benchmark["metrics"] = {}
            elif problem == "wrong-artifact":
                report["artifact"] = {"sha256": "0" * 64}
        return report

    def fake_recommend(hardware, constraints, **kwargs):
        fresh, = kwargs["evidence_store"].benchmarks
        assert fresh.id == f"fresh-{constraints.model_id}"
        return [SimpleNamespace(
            model=SimpleNamespace(id=constraints.model_id),
            benchmark=None if problem == "no-match" and constraints.model_id == "second" else fresh,
        )]

    monkeypatch.setattr("autonomyfit.deployment.validate_deployment", fake_validate)
    monkeypatch.setattr("autonomyfit.deployment.detect_hardware", _fake_hardware)
    monkeypatch.setattr(
        "autonomyfit.deployment.load_model_catalog",
        lambda **kwargs: SimpleNamespace(models=models),
    )
    monkeypatch.setattr("autonomyfit.deployment.recommend_models", fake_recommend)
    monkeypatch.setattr("autonomyfit.deployment.rank_recommendations", lambda items, objective: items)
    monkeypatch.setattr("autonomyfit.deployment.recommendation_dict", lambda item: {"model_id": item.model.id})
    result = assess_candidates(["first", "second"], artifacts, offline=True)
    assert len(result["reports"]) == 2
    assert result["reordered_recommendations"] == [{"model_id": "first"}]

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
