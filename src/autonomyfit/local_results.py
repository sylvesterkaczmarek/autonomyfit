from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark import hardware_evidence_id
from .models import HardwareProfile

LOCAL_RESULT_MAX_AGE_DAYS = 180


@dataclass(frozen=True)
class LocalResultStatus:
    path: Path
    benchmark_id: str | None
    model_id: str | None
    valid: bool
    reasons: tuple[str, ...]
    created_at: str | None
    runtime: str | None
    runtime_version: str | None
    artifact_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "benchmark_id": self.benchmark_id,
            "model_id": self.model_id,
            "valid": self.valid,
            "reasons": list(self.reasons),
            "created_at": self.created_at,
            "runtime": self.runtime,
            "runtime_version": self.runtime_version,
            "artifact_sha256": self.artifact_sha256,
        }


def _major(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(?<!\d)(\d+)(?:\.\d+)?", value)
    return int(match.group(1)) if match else None


def _runtime_capability_version(hardware: HardwareProfile, runtime: str | None) -> str | None:
    if not runtime:
        return None
    aliases = {"onnx": "onnxruntime", "onnxruntime": "onnxruntime"}
    target = aliases.get(runtime.casefold(), runtime.casefold())
    for capability in hardware.runtimes:
        if capability.name.casefold() == target:
            return capability.version
    return None


def local_report_compatibility(
    document: dict[str, Any],
    hardware: HardwareProfile,
    *,
    max_age_days: int = LOCAL_RESULT_MAX_AGE_DAYS,
    now: datetime | None = None,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    current = now or datetime.now(timezone.utc)
    created_raw = document.get("created_at")
    try:
        created = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError
        age_days = (current - created.astimezone(timezone.utc)).days
        if age_days > max_age_days:
            reasons.append(f"local result is stale ({age_days} days > {max_age_days})")
    except (TypeError, ValueError):
        reasons.append("local result timestamp is invalid")

    report_hardware = document.get("hardware") or {}
    report_id = report_hardware.get("id")
    current_id = hardware_evidence_id(hardware)
    if current_id and report_id and report_id != current_id:
        reasons.append(f"hardware identity changed ({report_id} != {current_id})")


    report_os = report_hardware.get("os")
    if report_os and hardware.os_name not in {"profile", str(report_os)}:
        reasons.append(f"operating-system identity changed ({report_os} -> {hardware.os_name})")

    report_driver = report_hardware.get("driver")
    if report_driver and hardware.driver:
        report_major = _major(str(report_driver))
        current_major = _major(hardware.driver)
        if report_major is not None and current_major is not None and report_major != current_major:
            reasons.append(
                f"driver major version changed ({report_driver} -> {hardware.driver})"
            )

    software = document.get("software") or {}

    provider = software.get("provider")
    provider_map = {
        "TensorrtExecutionProvider": "tensorrt-ep",
        "CUDAExecutionProvider": "cuda-ep",
        "OpenVINOExecutionProvider": "openvino-ep",
        "CoreMLExecutionProvider": "coreml-ep",
        "QNNExecutionProvider": "qnn",
        "XNNPACKExecutionProvider": "xnnpack",
        "VitisAIExecutionProvider": "vitisai",
    }
    provider_capability = provider_map.get(str(provider)) if provider else None
    if provider_capability:
        capability = next(
            (item for item in hardware.runtimes if item.name == provider_capability), None
        )
        if capability is None or not capability.available:
            reasons.append(f"execution provider is no longer available ({provider})")

    runtime = software.get("runtime")
    report_runtime_version = software.get("runtime_version")
    current_runtime_version = _runtime_capability_version(
        hardware, str(runtime) if runtime else None
    )
    if report_runtime_version and current_runtime_version:
        report_major = _major(str(report_runtime_version))
        current_major = _major(current_runtime_version)
        if report_major is not None and current_major is not None and report_major != current_major:
            reasons.append(
                "runtime major version changed "
                f"({report_runtime_version} -> {current_runtime_version})"
            )

    return not reasons, tuple(reasons)


def inspect_local_result(
    path: Path,
    hardware: HardwareProfile,
    *,
    max_age_days: int = LOCAL_RESULT_MAX_AGE_DAYS,
    now: datetime | None = None,
) -> LocalResultStatus:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return LocalResultStatus(
            path, None, None, False, ("report is unreadable or invalid JSON",), None, None, None, None
        )
    if not isinstance(value, dict):
        return LocalResultStatus(
            path, None, None, False, ("report root is not a JSON object",), None, None, None, None
        )
    valid, reasons = local_report_compatibility(
        value, hardware, max_age_days=max_age_days, now=now
    )
    software = value.get("software") or {}
    model = value.get("model") or {}
    artifact = value.get("artifact") or {}
    return LocalResultStatus(
        path=path,
        benchmark_id=value.get("benchmark_id"),
        model_id=model.get("id"),
        valid=valid,
        reasons=reasons,
        created_at=value.get("created_at"),
        runtime=software.get("runtime"),
        runtime_version=software.get("runtime_version"),
        artifact_sha256=artifact.get("sha256"),
    )


def list_local_results(
    directory: Path,
    hardware: HardwareProfile,
    *,
    max_age_days: int = LOCAL_RESULT_MAX_AGE_DAYS,
) -> list[LocalResultStatus]:
    if not directory.exists():
        return []
    return [
        inspect_local_result(path, hardware, max_age_days=max_age_days)
        for path in sorted(directory.glob("*.json"))
    ]
