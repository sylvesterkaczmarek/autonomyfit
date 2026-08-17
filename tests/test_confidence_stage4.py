from dataclasses import replace

from autonomyfit.evidence import (
    BenchmarkEvidence,
    EvidenceStore,
    LatencyStats,
    PowerStats,
)
from autonomyfit.hardware import hardware_from_profile
from autonomyfit.models import Constraints, RuntimeCapability
from autonomyfit.scoring import recommend_models


def _store(source_date: str, quality: str = "local-measured") -> EvidenceStore:
    evidence = BenchmarkEvidence(
        id="exact", model_id="yolo26n", model_revision="r1", artifact_id="a1",
        artifact_sha256="a" * 64, artifact_format="tensorrt-engine",
        hardware_id="jetson-orin-nx-16gb", hardware_name="Jetson Orin NX 16GB",
        runtime="tensorrt", runtime_version="10.0", provider="trtexec", precision="fp16",
        quantization=None, batch_size=1, input_shapes={"images":[1,3,640,640]}, power_mode="MAXN",
        clocks={}, warmup=10, iterations=100,
        latency=LatencyStats(mean_ms=4.0, median_ms=3.9, p50_ms=3.9, p90_ms=4.2, p95_ms=4.3, p99_ms=4.5),
        throughput_fps=250.0, power=PowerStats(mean_w=12.0, scope="Jetson VDD_IN rail"),
        peak_memory_mb=1200.0, peak_memory_scope="process RSS", quality=quality,
        source_id="local", source_url="local://benchmark", source_date=source_date,
        software_stack_id="stack-1", provider_version="10.0",
        machine_source="detected", verified_identity=True,
    )
    return EvidenceStore(document={}, benchmarks=(evidence,))


def _hardware():
    return replace(
        hardware_from_profile("jetson-orin-nx-16gb"),
        runtimes=(RuntimeCapability("tensorrt", True, "10.0", "local", verified=True),),
    )


def _constraints(**kwargs):
    values = {
        "task": "detection", "model_id": "yolo26n", "model_revision": "r1",
        "artifact_sha256": "a" * 64, "runtime": "tensorrt", "precision": "fp16",
        "provider": "trtexec", "provider_version": "10.0", "batch_size": 1,
        "input_shapes": {"images": [1, 3, 640, 640]}, "power_mode": "MAXN",
        "software_stack_id": "stack-1", "max_latency_ms": 5.0,
    }
    values.update(kwargs)
    return Constraints(**values)


def test_exact_fresh_local_evidence_has_high_numeric_confidence():
    item = recommend_models(
        _hardware(), _constraints(), offline=True, evidence_store=_store("2026-08-16")
    )[0]
    assert item.verdict == "VERIFIED_FIT"
    assert item.confidence is not None
    assert item.confidence.score >= 85
    assert item.confidence.level == "HIGH"


def test_stale_evidence_reduces_confidence_and_cannot_verify_fit():
    item = recommend_models(
        _hardware(), _constraints(), offline=True, evidence_store=_store("2020-01-01")
    )[0]
    assert item.verdict == "BENCHMARK_REQUIRED"
    assert item.confidence is not None
    assert item.confidence.score < 85


def test_unresolved_constraint_caps_confidence():
    item = recommend_models(
        hardware_from_profile("nvidia-t4-16gb"),
        Constraints(task="anomaly", max_power_w=20),
        offline=True,
    )[0]
    assert item.verdict == "BENCHMARK_REQUIRED"
    assert item.confidence is not None
    assert item.confidence.score <= 55
    assert item.next_benchmark

def test_stale_registry_caps_recommendation_confidence(monkeypatch):
    from autonomyfit.catalog import LoadedCatalog, load_model_catalog
    from autonomyfit.models import RegistryProvenance

    loaded = load_model_catalog(offline=True)
    model = next(item for item in loaded.models if item.id == "yolo26n")
    stale = LoadedCatalog(
        models=(model,),
        provenance=RegistryProvenance(source="cache", stale=True, signature_verified=True),
    )
    monkeypatch.setattr("autonomyfit.scoring.load_model_catalog", lambda *args, **kwargs: stale)
    item = recommend_models(
        _hardware(), _constraints(), offline=True, evidence_store=_store("2026-08-16")
    )[0]
    assert item.confidence is not None
    assert item.confidence.score <= 55
    assert "registry freshness" in item.confidence.unresolved_constraints
    assert "current registry freshness" in item.unknowns
