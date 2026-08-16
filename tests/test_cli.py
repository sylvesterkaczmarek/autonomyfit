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


def test_unknown_profile_is_reported_cleanly():
    result = runner.invoke(app, ["recommend", "--hardware-profile", "does-not-exist"])
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
    result = runner.invoke(app, ["recommend", "--fps", "-1"])
    assert result.exit_code != 0


def test_catalog_rejects_unknown_task():
    result = runner.invoke(app, ["catalog", "--task", "audio"])
    assert result.exit_code != 0
    assert "task must be detection or vlm" in result.output
