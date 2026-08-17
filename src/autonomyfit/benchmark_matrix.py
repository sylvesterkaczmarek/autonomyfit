from __future__ import annotations

import hashlib
import json
from typing import Any

from .evidence import BenchmarkEvidence, EvidenceStore, load_evidence_store


def matrix_key(evidence: BenchmarkEvidence) -> str:
    payload = {
        "model": evidence.model_id,
        "revision": evidence.model_revision,
        "artifact_sha256": evidence.artifact_sha256,
        "hardware": evidence.hardware_id,
        "runtime": evidence.runtime,
        "runtime_version": evidence.runtime_version,
        "provider": evidence.provider,
        "provider_version": evidence.provider_version,
        "precision": evidence.precision,
        "quantization": evidence.quantization,
        "batch_size": evidence.batch_size,
        "input_shapes": evidence.input_shapes,
        "power_mode": evidence.power_mode,
        "software_stack_id": evidence.software_stack_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return f"matrix-{digest}"


def _row(evidence: BenchmarkEvidence) -> dict[str, Any]:
    return {
        "matrix_key": matrix_key(evidence),
        "benchmark_id": evidence.id,
        "model_id": evidence.model_id,
        "model_revision": evidence.model_revision,
        "artifact_sha256": evidence.artifact_sha256,
        "hardware_id": evidence.hardware_id,
        "runtime": evidence.runtime,
        "runtime_version": evidence.runtime_version,
        "provider": evidence.provider,
        "provider_version": evidence.provider_version,
        "precision": evidence.precision,
        "quantization": evidence.quantization,
        "batch_size": evidence.batch_size,
        "input_shapes": evidence.input_shapes,
        "power_mode": evidence.power_mode,
        "software_stack_id": evidence.software_stack_id,
        "quality": evidence.quality,
        "verified_identity": evidence.verified_identity,
        "exact_context_complete": evidence.eligible_for_verified_fit,
        "latency_ms": evidence.latency_ms,
        "throughput_fps": evidence.fps,
        "power_mean_w": evidence.power.mean_w,
        "power_scope": evidence.power.scope,
        "peak_memory_mb": evidence.peak_memory_mb,
        "source_date": evidence.source_date,
    }


def benchmark_matrix(
    *,
    store: EvidenceStore | None = None,
    local_only: bool = False,
    model_id: str | None = None,
    hardware_id: str | None = None,
) -> dict[str, Any]:
    evidence_store = store or load_evidence_store(include_local=True)
    items = list(evidence_store.benchmarks)
    if local_only:
        items = [item for item in items if item.quality == "local-measured"]
    if model_id:
        items = [item for item in items if item.model_id.casefold() == model_id.casefold()]
    if hardware_id:
        items = [item for item in items if item.hardware_id.casefold() == hardware_id.casefold()]
    rows = [_row(item) for item in items]
    rows.sort(key=lambda item: (item["model_id"], item["hardware_id"], item["runtime"], item["matrix_key"]))
    return {
        "record_count": len(rows),
        "exact_context_complete": sum(bool(row["exact_context_complete"]) for row in rows),
        "local_measured": sum(row["quality"] == "local-measured" for row in rows),
        "vendor_published": sum(row["quality"] == "vendor-published" for row in rows),
        "standardized": sum(row["quality"] == "standardized" for row in rows),
        "matrix": rows,
    }
