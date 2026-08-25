from __future__ import annotations

from pathlib import Path
from typing import Any

from .deployment import (
    DeploymentValidationError,
    ValidationOptions,
    validate_deployment,
)
from .hardware import detect_hardware, hardware_from_profile
from .models import Constraints, Objective, Recommendation
from .scoring import recommend_models as _recommend_models


def recommend(
    task: str = "detection",
    *,
    hardware_profile: str | None = None,
    objective: Objective = "balanced",
    runtime: str | None = None,
    precision: str | None = None,
    min_fps: float | None = None,
    max_latency_ms: float | None = None,
    max_power_w: float | None = None,
    min_accuracy: float | None = None,
    max_memory_gb: float | None = None,
    max_params_m: float | None = None,
    min_confidence: float | None = None,
    verified_only: bool = False,
    include_experimental: bool = False,
    limit: int = 8,
    offline: bool = False,
) -> list[Recommendation]:
    """Recommend models using AutonomyFit's existing evidence and ranking engine."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    hardware = hardware_from_profile(hardware_profile) if hardware_profile else detect_hardware()
    constraints = Constraints(
        task=task,
        objective=objective,
        runtime=runtime,
        precision=precision,
        min_fps=min_fps,
        max_latency_ms=max_latency_ms,
        max_power_w=max_power_w,
        min_accuracy=min_accuracy,
        max_memory_gb=max_memory_gb,
        max_params_m=max_params_m,
        min_confidence=min_confidence,
        verified_only=verified_only,
        include_experimental=include_experimental,
    )
    return _recommend_models(hardware, constraints, offline=offline)[:limit]


def assess_deployment(
    model_id: str,
    *,
    artifact: str | Path | None = None,
    hardware_profile: str | None = None,
    runtime: str | None = None,
    precision: str | None = None,
    expected_sha256: str | None = None,
    trust_artifact: bool = False,
    offline: bool = False,
    max_latency_ms: float | None = None,
    min_fps: float | None = None,
    max_power_w: float | None = None,
    max_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Assess a model or local artifact with AutonomyFit's deployment validator.

    This intentionally does not expose remote acquisition, conversion, or benchmarking;
    use the CLI for those advanced workflows.
    """
    artifact_path = Path(artifact).expanduser().resolve() if artifact is not None else None
    options = ValidationOptions(
        model_id=model_id,
        artifact=artifact_path,
        hardware_profile=hardware_profile,
        runtime=runtime,
        precision=precision,
        expected_sha256=expected_sha256,
        trust_artifact=trust_artifact,
        offline=offline,
        max_latency_ms=max_latency_ms,
        min_fps=min_fps,
        max_power_w=max_power_w,
        max_memory_gb=max_memory_gb,
    )
    return validate_deployment(options)


__all__ = ["DeploymentValidationError", "assess_deployment", "recommend"]
