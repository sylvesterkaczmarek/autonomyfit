from __future__ import annotations

import pytest

from autonomyfit.deployment_reports import (
    DeploymentReportError,
    load_deployment_report,
    render_deployment_markdown,
    save_deployment_report,
    validate_deployment_report,
)


def _report():
    return {
        "schema_version": 1,
        "created_at": "2026-08-16T13:00:00Z",
        "status": "validated",
        "validation_scope": "identity-and-compatibility",
        "autonomyfit_version": "0.6.0",
        "registry": {"registry_version": 4, "source": "cache"},
        "machine": {
            "matched_profile": "jetson-orin-nx-16gb",
            "gpu": "Jetson Orin NX",
            "cpu": "Arm",
            "os_name": "Linux",
            "architecture": "aarch64",
        },
        "software_stack": {"python_version": "3.12", "autonomyfit_version": "0.6.0"},
        "constraints": {"max_latency_ms": 10.0},
        "model": {
            "id": "demo",
            "revision": "a" * 40,
            "license_spdx": "Apache-2.0",
            "license_status": "published",
        },
        "artifact": {
            "filename": "model.onnx",
            "sha256": "b" * 64,
            "source": "huggingface",
        },
        "runtime": {"name": "onnxruntime", "precision": "fp16", "available": True},
        "conversion": None,
        "compatibility": {"checks": [{"name": "onnx-structure", "status": "pass", "detail": "ok"}]},
        "benchmark": None,
        "registry_comparison": None,
        "recommendation": {"verdict": "FEASIBLE", "confidence": {"score": 72.0}, "blockers": []},
        "warnings": ["performance not measured"],
        "reproducibility": {
            "commands": [
                "autonomyfit validate demo --revision " + "a" * 40 + " --sha256 " + "b" * 64
            ]
        },
    }


def test_report_schema_and_readable_render_are_reproducible(tmp_path):
    report = _report()
    validate_deployment_report(report)
    text = render_deployment_markdown(report)
    assert "Artifact SHA-256" in text
    assert "72.000/100" in text
    assert "max_latency_ms" in text
    assert "autonomyfit validate demo" in text

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    save_deployment_report(report, json_path)
    save_deployment_report(report, md_path)
    assert load_deployment_report(json_path)["artifact"]["sha256"] == "b" * 64
    assert md_path.read_text().startswith("# AutonomyFit deployment report")


def test_invalid_report_is_rejected():
    report = _report()
    del report["runtime"]
    with pytest.raises(DeploymentReportError, match="runtime"):
        validate_deployment_report(report)
