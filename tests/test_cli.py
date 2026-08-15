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
