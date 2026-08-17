import json

import pytest
import typer
from typer.testing import CliRunner

from autonomyfit.cli import _parse_named_shapes, app

runner = CliRunner()


def test_scan_json_smoke():
    result = runner.invoke(app, ["scan", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "platform" in payload
    assert "ram_total_gb" in payload


def test_profile_recommendation_uses_contextual_vendor_evidence():
    result = runner.invoke(
        app,
        [
            "recommend",
            "--offline",
            "--hardware-profile",
            "jetson-orin-nx-16gb",
            "--fps",
            "200",
            "--latency-ms",
            "5",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    yolo26n = next(item for item in payload if item["model_id"] == "yolo26n")
    assert yolo26n["verdict"] == "BENCHMARK_REQUIRED"
    assert yolo26n["latency_ms"] == 4.13
    assert yolo26n["evidence_confidence"] == "LOW"
    assert yolo26n["benchmark_match"]["exact"] is False


def test_unknown_profile_is_reported_cleanly():
    result = runner.invoke(app, ["recommend", "--offline", "--hardware-profile", "does-not-exist"])
    assert result.exit_code != 0
    assert "Unknown hardware profile" in result.output
    assert "Traceback" not in result.output


def test_invalid_shape_is_reported_cleanly(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    result = runner.invoke(app, ["benchmark", str(model), "--shape", "1,3,nope,640"])
    assert result.exit_code != 0
    assert "shape must be comma-separated positive integers" in result.output
    assert "Traceback" not in result.output


def test_negative_fps_is_rejected():
    result = runner.invoke(app, ["recommend", "--offline", "--fps", "-1"])
    assert result.exit_code != 0


def test_catalog_rejects_unknown_task():
    result = runner.invoke(app, ["catalog", "--offline", "--task", "not-a-task"])
    assert result.exit_code != 0
    assert "unknown task" in result.output


def test_registry_status_json(tmp_path):
    result = runner.invoke(
        app,
        ["registry", "status", "--json"],
        env={"AUTONOMYFIT_CACHE_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["cache"] is None
    assert payload["fallback"]["registry_version"] == 4


def test_registry_clear_cache_preserves_security_state_message(tmp_path):
    result = runner.invoke(
        app,
        ["registry", "clear-cache", "--json"],
        env={"AUTONOMYFIT_CACHE_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["security_state_preserved"] is True


def test_benchmark_backends_json_smoke():
    result = runner.invoke(app, ["benchmark-backends", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert {item["name"] for item in payload} >= {"onnxruntime", "tensorrt", "openvino", "coreml"}


def test_artifact_matching_requires_model_id(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    result = runner.invoke(app, ["recommend", "--offline", "--artifact", str(artifact)])
    assert result.exit_code != 0
    assert "requires" in result.output
    assert "artifact" in result.output


def test_named_input_shapes_are_parsed_and_duplicates_rejected():
    assert _parse_named_shapes(["image=1,3,640,640", "mask=1,1,640,640"]) == {
        "image": [1, 3, 640, 640],
        "mask": [1, 1, 640, 640],
    }
    with pytest.raises(typer.BadParameter, match="duplicate"):
        _parse_named_shapes(["image=1,3,640,640", "image=1,3,224,224"])


def test_shape_and_named_input_shape_conflict_is_rejected(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(model),
            "--shape",
            "1,3,640,640",
            "--input-shape",
            "image=1,3,640,640",
        ],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.output
