import pytest

from autonomyfit.mlperf import MLPerfImportError, parse_mlperf_summary


def test_parse_mlperf_single_stream_latency():
    summary = parse_mlperf_summary(
        "Scenario : SingleStream\nResult is : VALID\n90th percentile latency (ns) : 4200000"
    )
    assert summary.valid is True
    assert summary.scenario == "SingleStream"
    assert summary.latency_ms == 4.2


def test_parse_mlperf_offline_throughput():
    summary = parse_mlperf_summary(
        "Scenario : Offline\nResult is : VALID\nSamples per second : 1234.5"
    )
    assert summary.throughput_fps == 1234.5


def test_mlperf_requires_validity_marker():
    with pytest.raises(MLPerfImportError):
        parse_mlperf_summary("Samples per second : 1")
