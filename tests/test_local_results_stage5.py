from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from autonomyfit.benchmark import hardware_evidence_id
from autonomyfit.local_results import local_report_compatibility
from autonomyfit.models import HardwareProfile, RuntimeCapability


def _hardware(*, driver="550.1", runtime_version="1.20.0", qnn=True):
    return HardwareProfile(
        platform="nvidia",
        os_name="Linux",
        architecture="x86_64",
        cpu="Demo CPU",
        ram_total_gb=32,
        ram_available_gb=24,
        gpu="Demo GPU",
        driver=driver,
        runtimes=(
            RuntimeCapability("onnxruntime", True, runtime_version, provider="CPUExecutionProvider"),
            RuntimeCapability("qnn", qnn, runtime_version if qnn else None, provider="QNNExecutionProvider", verified=False),
        ),
    )


def _report(hardware, *, created="2026-08-16T12:00:00Z", runtime_version="1.20.0", provider="CPUExecutionProvider"):
    return {
        "benchmark_id": "local-12345678",
        "created_at": created,
        "model": {"id": "demo"},
        "artifact": {"sha256": "a" * 64},
        "hardware": {
            "id": hardware_evidence_id(hardware),
            "os": "Linux",
            "driver": "550.1",
        },
        "software": {
            "runtime": "onnxruntime",
            "runtime_version": runtime_version,
            "provider": provider,
        },
    }


def test_current_exact_machine_result_is_valid():
    hardware = _hardware()
    valid, reasons = local_report_compatibility(
        _report(hardware), hardware, now=datetime(2026, 8, 16, 14, tzinfo=timezone.utc)
    )
    assert valid is True
    assert reasons == ()


def test_local_result_expires_and_invalidates_on_major_stack_changes():
    hardware = _hardware()
    stale, reasons = local_report_compatibility(
        _report(hardware, created="2025-01-01T00:00:00Z"),
        hardware,
        now=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
    assert stale is False
    assert any("stale" in reason for reason in reasons)

    valid, reasons = local_report_compatibility(
        _report(hardware), _hardware(driver="600.1"), now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("driver major" in reason for reason in reasons)

    valid, reasons = local_report_compatibility(
        _report(hardware), _hardware(runtime_version="2.0"), now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("runtime major" in reason for reason in reasons)


def test_provider_disappearance_invalidates_bridge_result():
    hardware = _hardware(qnn=True)
    document = _report(hardware, provider="QNNExecutionProvider")
    valid, reasons = local_report_compatibility(
        document, _hardware(qnn=False), now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("no longer available" in reason for reason in reasons)

def test_future_dated_local_result_is_invalid():
    hardware = _hardware()
    future = _report(hardware, created="2099-01-01T00:00:00Z")
    valid, reasons = local_report_compatibility(
        future, hardware, now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("future" in reason for reason in reasons)

def test_power_mode_change_invalidates_local_result():
    from dataclasses import replace

    hardware = replace(_hardware(), power_mode="MODE_15W")
    report = _report(hardware)
    report["hardware"]["power_mode"] = "MODE_30W"
    valid, reasons = local_report_compatibility(
        report, hardware, now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("power mode" in reason for reason in reasons)


def test_native_runtime_disappearance_invalidates_local_result():
    hardware = _hardware()
    document = _report(hardware)
    document["software"].update(
        {"runtime": "openvino", "runtime_version": "2026.3.0", "provider": "CPU"}
    )
    valid, reasons = local_report_compatibility(
        document, hardware, now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("runtime is no longer available" in reason for reason in reasons)


def test_unknown_current_runtime_version_invalidates_exact_local_context():
    hardware = _hardware()
    unknown = HardwareProfile(
        platform=hardware.platform,
        os_name=hardware.os_name,
        architecture=hardware.architecture,
        cpu=hardware.cpu,
        ram_total_gb=hardware.ram_total_gb,
        ram_available_gb=hardware.ram_available_gb,
        gpu=hardware.gpu,
        driver=hardware.driver,
        runtimes=(RuntimeCapability("onnxruntime", True, None, provider="CPUExecutionProvider"),),
    )
    document = _report(unknown)
    valid, reasons = local_report_compatibility(
        document, unknown, now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("runtime version could not be established" in reason for reason in reasons)


@pytest.mark.parametrize("section", ["hardware", "software", "reproducibility"])
@pytest.mark.parametrize("value", [["invalid"], "invalid", 42])
def test_malformed_local_context_is_rejected_without_crashing(section, value):
    report = _report(_hardware())
    report[section] = value
    valid, reasons = local_report_compatibility(report, _hardware())
    assert not valid
    assert any(section in reason for reason in reasons)


@pytest.mark.parametrize("section", ["model", "software", "artifact", "hardware"])
def test_inspection_reports_malformed_nested_objects(tmp_path, section):
    from autonomyfit.local_results import inspect_local_result

    report = _report(_hardware())
    report[section] = ["invalid"]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = inspect_local_result(path, _hardware())
    assert not result.valid
    assert "validation failed" in result.reasons[0]


def test_inspection_reports_invalid_utf8(tmp_path):
    from autonomyfit.local_results import inspect_local_result

    path = tmp_path / "report.json"
    path.write_bytes(b"\xff")
    result = inspect_local_result(path, _hardware())
    assert not result.valid
    assert "unreadable" in result.reasons[0]


@pytest.mark.parametrize("section, field", [("hardware", "id"), ("software", "runtime")])
def test_local_compatibility_requires_hardware_and_runtime_identity(section, field):
    report = _report(_hardware())
    del report[section][field]
    valid, reasons = local_report_compatibility(
        report, _hardware(), now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert not valid
    assert any("identity" in reason for reason in reasons)
