from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


class DeploymentReportError(RuntimeError):
    """Deployment report is invalid or cannot be rendered."""


def _schema_path() -> Path:
    from importlib.resources import files

    return Path(str(files("autonomyfit.data").joinpath("deployment-report-v1.schema.json")))


def validate_deployment_report(document: dict[str, Any]) -> None:
    from importlib.resources import files

    schema = json.loads(
        files("autonomyfit.data")
        .joinpath("deployment-report-v1.schema.json")
        .read_text(encoding="utf-8")
    )
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda item: list(item.path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "document"
        raise DeploymentReportError(
            f"deployment report schema error at {location}: {first.message}"
        )


def save_deployment_report(document: dict[str, Any], path: Path) -> None:
    validate_deployment_report(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.casefold() in {".md", ".markdown"}:
        path.write_text(render_deployment_markdown(document), encoding="utf-8")
    else:
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def load_deployment_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentReportError(f"could not read deployment report: {exc}") from exc
    if not isinstance(value, dict):
        raise DeploymentReportError("deployment report must be a JSON object")
    validate_deployment_report(value)
    return value


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def render_deployment_markdown(document: dict[str, Any]) -> str:
    validate_deployment_report(document)
    model = document["model"]
    artifact = document.get("artifact") or {}
    runtime = document["runtime"]
    machine = document["machine"]
    compatibility = document["compatibility"]
    software_stack = document.get("software_stack") or {}
    constraints = document.get("constraints") or {}
    conversion = document.get("conversion") or {}
    benchmark = document.get("benchmark") or {}
    metrics = benchmark.get("metrics") or {}
    latency = metrics.get("latency") or {}
    power = metrics.get("power") or {}
    comparison = document.get("registry_comparison") or {}
    recommendation = document.get("recommendation") or {}
    lines = [
        f"# AutonomyFit deployment report for {model['id']}",
        "",
        f"Generated: `{document['created_at']}`  ",
        f"Status: **{document['status']}**  ",
        f"Registry: v{document['registry'].get('registry_version', '?')} ({document['registry'].get('source', 'unknown')})",
        "",
        "## Identity",
        "",
        f"- Model: `{model['id']}`",
        f"- Revision: `{model.get('revision') or 'unresolved'}`",
        f"- Licence: `{model.get('license_spdx') or 'unknown'}` ({model.get('license_status', 'unknown')})",
        f"- Artifact: `{artifact.get('filename') or 'not selected'}`",
        f"- Artifact SHA-256: `{artifact.get('sha256') or 'unknown'}`",
        f"- Artifact source: `{artifact.get('source') or 'unknown'}`",
        "",
        "## Target",
        "",
        f"- Hardware: `{machine.get('matched_profile') or machine.get('gpu') or machine.get('cpu')}`",
        f"- OS: `{machine.get('os_name')}`",
        f"- Architecture: `{machine.get('architecture')}`",
        f"- Runtime: `{runtime.get('name')}`",
        f"- Runtime available: `{runtime.get('available')}`",
        f"- Precision: `{runtime.get('precision')}`",
        f"- Python: `{software_stack.get('python_version', 'unknown')}`",
        f"- AutonomyFit: `{software_stack.get('autonomyfit_version', document.get('autonomyfit_version'))}`",
        "",
        "## Compatibility",
        "",
    ]
    for check in compatibility.get("checks", []):
        marker = "PASS" if check.get("status") == "pass" else (
            "FAIL" if check.get("status") == "fail" else "INFO"
        )
        lines.append(f"- **{marker}** {check.get('name')}: {check.get('detail')}")
    if conversion:
        lines += [
            "",
            "## Conversion",
            "",
            f"- Path: `{conversion.get('source_format', 'unknown')} -> {conversion.get('target_format', 'unknown')}`",
            f"- Tool: `{conversion.get('tool', 'unknown')}`",
            f"- Target SHA-256: `{conversion.get('target_sha256', 'unknown')}`",
            f"- Equivalence: `{(conversion.get('equivalence') or {}).get('status', 'not-run')}`",
        ]
        for warning in conversion.get("warnings", []):
            lines.append(f"- Warning: {warning}")
    if any(value is not None for value in constraints.values()):
        lines += ["", "## Requested constraints", ""]
        for key, value in constraints.items():
            if value is not None:
                lines.append(f"- {key}: `{value}`")

    if metrics:
        lines += [
            "",
            "## Local benchmark",
            "",
            f"- Median latency: {_fmt(latency.get('median_ms'), ' ms')}",
            f"- P95 latency: {_fmt(latency.get('p95_ms'), ' ms')}",
            f"- Throughput: {_fmt(metrics.get('throughput_fps'), ' FPS')}",
            f"- Peak process RSS: {_fmt(metrics.get('peak_memory_mb'), ' MB')}",
            f"- Mean power: {_fmt(power.get('mean_w'), ' W')}",
            f"- Energy: {_fmt(power.get('energy_j'), ' J')}",
        ]
    if comparison:
        lines += [
            "",
            "## Registry comparison",
            "",
            f"- Classification: `{comparison.get('classification', 'not-comparable')}`",
            f"- Expected latency: {_fmt(comparison.get('expected_latency_ms'), ' ms')}",
            f"- Expected range: `{comparison.get('expected_range_ms') or 'unknown'}`",
            f"- Local latency: {_fmt(comparison.get('local_latency_ms'), ' ms')}",
        ]
        for warning in comparison.get("warnings", []):
            lines.append(f"- Warning: {warning}")
    if recommendation:
        lines += [
            "",
            "## Constraint result",
            "",
            f"- Verdict: **{recommendation.get('verdict', 'unknown')}**",
            f"- Confidence: {_fmt((recommendation.get('confidence') or {}).get('score'))}/100",
        ]
        for blocker in recommendation.get("blockers", []):
            lines.append(f"- Failed: {blocker}")
    warnings = document.get("warnings") or []
    if warnings:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {warning}" for warning in warnings)
    lines += ["", "## Reproduce", ""]
    for command in document["reproducibility"].get("commands", []):
        lines += ["```bash", command, "```", ""]
    return "\n".join(lines).rstrip() + "\n"
