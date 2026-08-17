import json
from datetime import date

import pytest

from autonomyfit.evidence import (
    BenchmarkEvidence,
    EvidenceSchemaError,
    LatencyStats,
    PowerStats,
    benchmark_evidence_from_report,
    import_benchmark_report,
    load_evidence_store,
    match_benchmarks,
    validate_benchmark_report,
)


def _evidence(**overrides):
    values = {
        "id": "bench",
        "model_id": "model",
        "model_revision": "revision-1",
        "artifact_id": "artifact",
        "artifact_sha256": "a" * 64,
        "artifact_format": "onnx",
        "hardware_id": "hardware",
        "hardware_name": "Hardware",
        "runtime": "onnx",
        "runtime_version": "1.0",
        "provider": "CPUExecutionProvider",
        "precision": "fp16",
        "quantization": None,
        "batch_size": 1,
        "input_shapes": {"input": [1, 3, 640, 640]},
        "power_mode": None,
        "clocks": {},
        "warmup": 10,
        "iterations": 50,
        "latency": LatencyStats(mean_ms=5.0),
        "throughput_fps": 200.0,
        "power": PowerStats(),
        "peak_memory_mb": 100.0,
        "peak_memory_scope": "process RSS",
        "quality": "local-measured",
        "source_id": "local",
        "source_url": "local://benchmark",
        "source_date": "2026-08-16",
        "software_stack_id": "stack-1",
        "provider_version": "1.0",
        "machine_source": "detected",
        "verified_identity": True,
    }
    values.update(overrides)
    return BenchmarkEvidence(**values)


def _match(evidence, **overrides):
    values = {
        "model_id": "model",
        "model_revision": "revision-1",
        "hardware_id": "hardware",
        "runtime": "onnx",
        "precision": "fp16",
        "artifact_sha256": "a" * 64,
        "runtime_version": "1.0",
        "provider": "CPUExecutionProvider",
        "provider_version": "1.0",
        "batch_size": 1,
        "input_shapes": {"input": [1, 3, 640, 640]},
        "software_stack_id": "stack-1",
        "today": date(2026, 8, 16),
    }
    values.update(overrides)
    return match_benchmarks([evidence], **values)


def test_exact_identity_match_can_be_verified():
    match = _match(_evidence())[0]
    assert match.exact is True
    assert match.evidence.eligible_for_verified_fit is True


def test_revision_mismatch_is_rejected():
    assert _match(_evidence(), model_revision="different") == []


def test_artifact_hash_mismatch_is_rejected():
    assert _match(_evidence(), artifact_sha256="b" * 64) == []


def test_runtime_precision_hardware_mismatch_is_rejected():
    assert _match(_evidence(), runtime="tensorrt") == []
    assert _match(_evidence(), precision="int8") == []
    assert _match(_evidence(), hardware_id="other") == []


def test_runtime_version_mismatch_is_rejected():
    assert _match(_evidence(), runtime_version="2.0") == []


def test_vendor_evidence_is_contextual_even_with_identity():
    evidence = _evidence(quality="vendor-published")
    match = _match(evidence)[0]
    assert match.exact is True
    assert evidence.eligible_for_verified_fit is False


def test_stale_evidence_is_not_exact():
    match = _match(_evidence(source_date="2020-01-01"))[0]
    assert match.exact is False
    assert any("older" in reason for reason in match.reasons)


def test_missing_identity_is_contextual():
    evidence = _evidence(model_revision=None, artifact_sha256=None, verified_identity=False)
    match = _match(evidence, model_revision=None, artifact_sha256=None, runtime_version="1.0")[0]
    assert match.exact is False
    assert match.identity_complete is False


def _report():
    return {
        "schema_version": 2,
        "benchmark_id": "local-12345678",
        "created_at": "2026-08-16T10:00:00Z",
        "quality": "local-measured",
        "notes": None,
        "model": {"id": "model", "revision": "revision-1"},
        "artifact": {"path": "model.onnx", "format": "onnx", "sha256": "a" * 64, "size_bytes": 10},
        "hardware": {
            "id": "hardware", "platform": "cpu", "device": None, "cpu": "CPU",
            "ram_total_gb": 8, "os": "Linux", "architecture": "x86_64", "driver": None,
            "power_mode": None, "clocks": {}, "thermal_c": {},
        },
        "software": {
            "runtime": "onnx", "runtime_version": "1.0", "provider": "CPUExecutionProvider",
            "provider_version": "1.0", "python_version": "3.12", "autonomyfit_version": "0.4.0",
        },
        "execution": {
            "precision": "fp16", "quantization": None, "batch_size": 1,
            "input_shapes": {"input": [1, 3, 640, 640]}, "warmup": 10, "iterations": 50,
            "random_seed": 0, "backend_options": {},
        },
        "metrics": {
            "load_ms": 10.0,
            "latency": {
                "min_ms": 4.0, "mean_ms": 5.0, "median_ms": 4.9, "p50_ms": 4.9,
                "p90_ms": 5.5, "p95_ms": 5.7, "p99_ms": 6.0, "max_ms": 6.2, "stdev_ms": 0.4,
            },
            "throughput_fps": 200.0, "peak_memory_mb": 150.0, "peak_memory_scope": "process RSS",
            "power": {"mean_w": 12.0, "max_w": 14.0, "energy_j": 6.0, "scope": "Jetson VDD_IN rail"},
        },
        "measurement": {"machine_source": "detected", "profile_only": False, "artifact_identity_verified": True},
        "reproducibility": {
            "command": "autonomyfit benchmark",
            "hostname_hash": "abc",
            "environment_fingerprint": "12345678",
            "software_stack_fingerprint": "1" * 64,
        },
    }


def test_benchmark_report_schema_and_conversion():
    report = _report()
    validate_benchmark_report(report)
    evidence = benchmark_evidence_from_report(report)
    assert evidence.eligible_for_verified_fit
    assert evidence.power.scope == "Jetson VDD_IN rail"


def test_malformed_benchmark_report_is_rejected():
    report = _report()
    report["artifact"]["sha256"] = "bad"
    with pytest.raises(EvidenceSchemaError):
        validate_benchmark_report(report)


def test_import_validates_before_copy(tmp_path):
    source = tmp_path / "report.json"
    source.write_text(json.dumps(_report()))
    target = import_benchmark_report(source, tmp_path / "store")
    assert target.exists()
    assert json.loads(target.read_text())["benchmark_id"] == "local-12345678"


def test_bundled_vendor_evidence_is_normalized_and_not_verified_fit():
    store = load_evidence_store(include_local=False)
    assert len(store.benchmarks) == 20
    assert all(item.quality == "vendor-published" for item in store.benchmarks)
    assert not any(item.eligible_for_verified_fit for item in store.benchmarks)

def test_benchmark_import_rejects_path_escaping_id(tmp_path):
    report = _report()
    report["benchmark_id"] = "../../outside-report"
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report))
    with pytest.raises(EvidenceSchemaError):
        import_benchmark_report(source, tmp_path / "store")
    assert not (tmp_path / "outside-report.json").exists()


def test_future_dated_benchmark_report_is_rejected():
    report = _report()
    report["created_at"] = "2099-01-01T00:00:00Z"
    with pytest.raises(EvidenceSchemaError, match="future"):
        validate_benchmark_report(report)

def test_local_evidence_requires_full_execution_context_for_exact_match():
    evidence = _evidence()
    assert _match(evidence, batch_size=2) == []
    assert _match(evidence, input_shapes={"input": [1, 3, 224, 224]}) == []
    assert _match(evidence, provider_version="2.0") == []
    assert _match(evidence, software_stack_id="stack-2") == []

    contextual = _match(evidence, software_stack_id=None)[0]
    assert contextual.exact is False
    assert any("software stack" in reason for reason in contextual.reasons)


def test_profile_only_local_report_is_rejected():
    report = _report()
    report["measurement"] = {
        "machine_source": "profile",
        "profile_only": True,
        "artifact_identity_verified": True,
    }
    with pytest.raises(EvidenceSchemaError, match="profile-only"):
        validate_benchmark_report(report)


def test_false_artifact_identity_flag_cannot_be_verified_fit():
    report = _report()
    report["measurement"]["artifact_identity_verified"] = False
    validate_benchmark_report(report)
    evidence = benchmark_evidence_from_report(report)
    assert evidence.verified_identity is False
    assert evidence.eligible_for_verified_fit is False


def test_local_report_requires_explicit_measurement_classification():
    report = _report()
    report.pop("measurement")
    with pytest.raises(EvidenceSchemaError):
        validate_benchmark_report(report)
