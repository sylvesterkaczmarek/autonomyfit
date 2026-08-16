from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .benchmark import sha256_file
from .evidence import BenchmarkEvidence, LatencyStats, PowerStats


class MLPerfImportError(ValueError):
    """MLPerf result cannot be represented as exact AutonomyFit evidence."""


@dataclass(frozen=True)
class MLPerfSummary:
    valid: bool
    scenario: str | None
    latency_ms: float | None
    throughput_fps: float | None


def parse_mlperf_summary(text: str) -> MLPerfSummary:
    valid_match = re.search(r"Result is\s*:\s*(VALID|INVALID)", text, re.IGNORECASE)
    if not valid_match:
        raise MLPerfImportError("MLPerf summary does not contain a result validity marker")
    valid = valid_match.group(1).upper() == "VALID"
    scenario_match = re.search(r"Scenario\s*:\s*([A-Za-z]+)", text, re.IGNORECASE)
    scenario = scenario_match.group(1) if scenario_match else None

    latency_ms = None
    # LoadGen reports latency in ns in official summaries.
    for label in ("90th percentile latency", "99th percentile latency", "Mean latency"):
        match = re.search(rf"{label}\s*\(ns\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if match:
            latency_ms = float(match.group(1)) / 1_000_000.0
            break

    throughput = None
    match = re.search(r"Samples per second\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        throughput = float(match.group(1))
    return MLPerfSummary(valid, scenario, latency_ms, throughput)


def import_mlperf_evidence(
    *,
    summary_path: Path,
    artifact_path: Path,
    model_id: str,
    model_revision: str,
    hardware_id: str,
    hardware_name: str,
    runtime: str,
    runtime_version: str | None,
    precision: str,
    source_url: str,
    source_date: str,
    mlperf_benchmark: str,
    expected_mlperf_benchmark: str,
) -> BenchmarkEvidence:
    """Create standardized evidence only when benchmark and identity are explicit.

    AutonomyFit intentionally requires the caller to assert the exact MLPerf benchmark
    expected for the registry model. This prevents a similarly named model family from
    inheriting a standardized result from a different MLPerf workload.
    """
    if mlperf_benchmark.casefold() != expected_mlperf_benchmark.casefold():
        raise MLPerfImportError(
            f"MLPerf benchmark {mlperf_benchmark!r} does not match expected "
            f"{expected_mlperf_benchmark!r} for {model_id}"
        )
    if not model_revision.strip():
        raise MLPerfImportError("MLPerf standardized evidence requires an exact model revision")
    summary = parse_mlperf_summary(summary_path.read_text(encoding="utf-8"))
    if not summary.valid:
        raise MLPerfImportError("MLPerf result is not VALID")
    artifact_hash = sha256_file(artifact_path)
    if summary.latency_ms is None and summary.throughput_fps is None:
        raise MLPerfImportError("MLPerf summary contains no supported performance metric")
    latency = LatencyStats(
        p90_ms=summary.latency_ms if summary.scenario and summary.scenario.casefold() == "singlestream" else None,
        p99_ms=summary.latency_ms if summary.scenario and summary.scenario.casefold() in {"server", "multistream"} else None,
        mean_ms=summary.latency_ms if summary.scenario is None else None,
    )
    return BenchmarkEvidence(
        id=f"mlperf:{mlperf_benchmark}:{artifact_hash[:16]}:{hardware_id}",
        model_id=model_id,
        model_revision=model_revision,
        artifact_id=f"mlperf-artifact:{artifact_hash[:20]}",
        artifact_sha256=artifact_hash,
        artifact_format=artifact_path.suffix.lstrip(".") or "unknown",
        hardware_id=hardware_id,
        hardware_name=hardware_name,
        runtime=runtime,
        runtime_version=runtime_version,
        provider="MLPerf LoadGen",
        precision=precision,
        quantization=None,
        batch_size=None,
        input_shapes={},
        power_mode=None,
        clocks={},
        warmup=None,
        iterations=None,
        latency=latency,
        throughput_fps=summary.throughput_fps,
        power=PowerStats(),
        peak_memory_mb=None,
        peak_memory_scope=None,
        quality="standardized",
        source_id=f"mlperf:{mlperf_benchmark}",
        source_url=source_url,
        source_date=source_date,
        software_stack_id=None,
        notes=f"Imported VALID MLPerf {mlperf_benchmark} {summary.scenario or 'unknown'} result.",
        verified_identity=True,
    )
