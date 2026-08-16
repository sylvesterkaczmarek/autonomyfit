import json

from typer.testing import CliRunner

from autonomyfit.cli import app

runner = CliRunner()


def test_scan_json_smoke():
    result = runner.invoke(app, ["scan", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "platform" in payload
    assert "ram_total_gb" in payload


def test_profile_recommendation_smoke():
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
    assert yolo26n["verdict"] == "VERIFIED_FIT"
    assert yolo26n["registry"]["source"] in {"cache", "bundled-fallback"}
    assert yolo26n["model_provenance"]["source_url"]


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
    result = runner.invoke(app, ["catalog", "--offline", "--task", "audio"])
    assert result.exit_code != 0
    assert "task must be detection or vlm" in result.output


def test_registry_status_json(tmp_path):
    result = runner.invoke(
        app,
        ["registry", "status", "--json"],
        env={"AUTONOMYFIT_CACHE_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["cache"] is None
    assert payload["fallback"]["registry_version"] == 1


def test_registry_clear_cache_preserves_security_state_message(tmp_path):
    result = runner.invoke(
        app,
        ["registry", "clear-cache", "--json"],
        env={"AUTONOMYFIT_CACHE_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["security_state_preserved"] is True
