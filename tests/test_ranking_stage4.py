from autonomyfit.evidence import BenchmarkEvidence, LatencyStats, PowerStats
from autonomyfit.models import (
    AccuracyMetric,
    ModelProfile,
    Recommendation,
)
from autonomyfit.ranking import dominates, pareto_layers, rank_recommendations


def _model(mid: str, *, accuracy: float = 70.0, params: float = 10.0) -> ModelProfile:
    return ModelProfile(
        id=mid,
        display_name=mid,
        family="test",
        task="classification",
        params_m=params,
        source_id="test",
        source_url="https://example.com/model",
        runtimes=("onnx",),
        accuracy=AccuracyMetric("Top-1", accuracy, "ImageNet"),
        supported_precisions=("fp32",),
    )


def _benchmark(mid: str, latency: float, fps: float, power: float | None = None):
    return BenchmarkEvidence(
        id=f"bench-{mid}", model_id=mid, model_revision="r", artifact_id="a",
        artifact_sha256="a" * 64, artifact_format="onnx", hardware_id="h",
        hardware_name="h", runtime="onnx", runtime_version="1", provider="CPU",
        precision="fp32", quantization=None, batch_size=1, input_shapes={}, power_mode=None,
        clocks={}, warmup=1, iterations=5, latency=LatencyStats(mean_ms=latency, median_ms=latency),
        throughput_fps=fps, power=PowerStats(mean_w=power), peak_memory_mb=None,
        peak_memory_scope=None, quality="local-measured", source_id="local",
        source_url="local://benchmark", source_date="2026-08-16", software_stack_id=None,
        verified_identity=True,
    )


def _rec(mid: str, *, latency=10.0, fps=100.0, accuracy=70.0, memory=1.0, power=None):
    benchmark = _benchmark(mid, latency, fps, power)
    return Recommendation(
        model=_model(mid, accuracy=accuracy), verdict="VERIFIED_FIT", score=0.0,
        runtime="onnx", precision="fp32", estimated_memory_gb=memory,
        memory_evidence="estimate", benchmark=benchmark, evidence_match=None,
        evidence_confidence="HIGH", runtime_available=True, reasons=(), blockers=(),
    )


def test_pareto_dominance_requires_same_known_metric_set():
    a = _rec("a", latency=5, fps=200, accuracy=80, memory=0.5, power=5)
    b = _rec("b", latency=10, fps=100, accuracy=70, memory=1.0, power=10)
    assert dominates(a, b)
    b_without_power = _rec("b2", latency=10, fps=100, accuracy=70, memory=1.0, power=None)
    assert not dominates(a, b_without_power)


def test_pareto_frontier_is_deterministic():
    a = _rec("a", latency=5, fps=200, accuracy=80, memory=0.5, power=5)
    b = _rec("b", latency=10, fps=100, accuracy=70, memory=1.0, power=10)
    c = _rec("c", latency=4, fps=180, accuracy=75, memory=0.8, power=7)
    ranks, dominates_map, dominated_by = pareto_layers([c, b, a])
    assert ranks["a"] == 0
    assert ranks["c"] == 0
    assert ranks["b"] > 0
    assert "b" in dominates_map["a"]
    assert "a" in dominated_by["b"]


def test_objectives_change_order_and_scores_are_normalized():
    fast = _rec("fast", latency=3, fps=300, accuracy=70, memory=2.0, power=12)
    accurate = _rec("accurate", latency=8, fps=120, accuracy=90, memory=1.0, power=8)
    latency = rank_recommendations([accurate, fast], "latency")
    accuracy = rank_recommendations([accurate, fast], "accuracy")
    assert latency[0].model.id == "fast"
    assert accuracy[0].model.id == "accurate"
    assert {item.score for item in latency} <= {0.0, 100.0}


def test_tie_breaking_uses_model_id():
    first = _rec("aaa")
    second = _rec("bbb")
    ordered = rank_recommendations([second, first], "balanced")
    assert [item.model.id for item in ordered] == ["aaa", "bbb"]
