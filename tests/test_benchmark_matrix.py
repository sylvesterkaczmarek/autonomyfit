from autonomyfit.benchmark_matrix import benchmark_matrix, matrix_key
from autonomyfit.evidence import (
    BenchmarkEvidence,
    EvidenceStore,
    LatencyStats,
    PowerStats,
)


def _evidence(**overrides):
    values = {
        "id": "local-a", "model_id": "demo", "model_revision": "r1", "artifact_id": "a",
        "artifact_sha256": "a" * 64, "artifact_format": "onnx", "hardware_id": "local-cpu-a",
        "hardware_name": "CPU", "runtime": "onnxruntime", "runtime_version": "1.20",
        "provider": "CPUExecutionProvider", "precision": "fp32", "quantization": None, "batch_size": 1,
        "input_shapes": {"input": [1, 4]}, "power_mode": None, "clocks": {}, "warmup": 2, "iterations": 5,
        "latency": LatencyStats(mean_ms=1.0, median_ms=1.0), "throughput_fps": 1000.0,
        "power": PowerStats(), "peak_memory_mb": 10, "peak_memory_scope": "process RSS",
        "quality": "local-measured", "source_id": "local", "source_url": "local://benchmark",
        "source_date": "2026-08-17", "software_stack_id": "stack-a",
        "provider_version": "onnxruntime-1.20", "machine_source": "detected", "verified_identity": True,
    }
    values.update(overrides)
    return BenchmarkEvidence(**values)


def test_matrix_key_changes_for_material_execution_context():
    base = _evidence()
    assert matrix_key(base) != matrix_key(_evidence(id="b", batch_size=2))
    assert matrix_key(base) != matrix_key(_evidence(id="c", input_shapes={"input": [1, 8]}))
    assert matrix_key(base) != matrix_key(_evidence(id="d", software_stack_id="stack-b"))


def test_matrix_summary_reports_exact_context():
    evidence = _evidence()
    payload = benchmark_matrix(
        store=EvidenceStore(document={}, benchmarks=(evidence,)), local_only=True
    )
    assert payload["record_count"] == 1
    assert payload["local_measured"] == 1
    assert payload["exact_context_complete"] == 1
