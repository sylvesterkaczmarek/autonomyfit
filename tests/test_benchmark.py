import pytest

from autonomyfit.benchmark import (
    _read_ina3221_vdd_in_power_w,
    _shape_for_input,
    latency_summary,
    parse_nvidia_smi_power_w,
    parse_tegrastats_power_w,
    percentile,
)


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_percentile_rejects_empty_input():
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_latency_summary_includes_tail_percentiles():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["median_ms"] == 3.0
    assert summary["p90_ms"] > summary["median_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]
    assert summary["stdev_ms"] > 0


def test_parse_jetson_total_power():
    assert parse_tegrastats_power_w("VDD_IN 12450mW/13000mW") == 12.45
    assert parse_tegrastats_power_w("VIN_SYS_5V0 9876mW/9000mW") is None
    assert parse_tegrastats_power_w("RAM 100/1000MB") is None


def test_read_jetson_vdd_in_from_ina3221(tmp_path):
    (tmp_path / "in1_label").write_text("VDD_IN\n")
    (tmp_path / "in1_input").write_text("5000\n")
    (tmp_path / "curr1_input").write_text("2400\n")
    assert _read_ina3221_vdd_in_power_w(tmp_path) == 12.0


def test_parse_nvidia_smi_power():
    assert parse_nvidia_smi_power_w("72.45") == 72.45
    assert parse_nvidia_smi_power_w("N/A") is None


def test_shape_override_must_match_input_rank():
    with pytest.raises(ValueError, match="expects 4"):
        _shape_for_input([1, 3, 640, 640], [1, 640, 640])
