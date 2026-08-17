from __future__ import annotations

import pytest

from autonomyfit.conversions import (
    ConversionError,
    convert_artifact,
    convert_to_tensorrt,
    convert_trusted_torchscript,
)


def test_tensorrt_conversion_fails_cleanly_when_tooling_is_unavailable(monkeypatch, tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    monkeypatch.setattr("autonomyfit.conversions.shutil.which", lambda name: None)
    with pytest.raises(ConversionError, match="trtexec"):
        convert_to_tensorrt(source, tmp_path / "out")


def test_pytorch_serialization_is_never_loaded_without_explicit_trust(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"not really torch")
    with pytest.raises(ConversionError, match="explicit trust"):
        convert_trusted_torchscript(
            source,
            "onnx",
            tmp_path / "out",
            input_shape=[1, 3, 32, 32],
            trust_source=False,
        )


def test_generic_onnx_to_coreml_is_refused_without_trusted_source_contract(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    with pytest.raises(ConversionError, match="intentionally not automated"):
        convert_artifact(source, "coreml", tmp_path / "out")


def test_identity_conversion_preserves_artifact_digest(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    result = convert_artifact(source, "onnxruntime", tmp_path / "out", precision="fp32")
    assert result.source_sha256 == result.target_sha256
    assert result.target_path == source
    assert result.built_locally is False

def test_conversion_rejects_source_identity_mismatch(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    with pytest.raises(ConversionError, match="changed before conversion"):
        convert_artifact(
            source,
            "onnxruntime",
            tmp_path / "out",
            precision="fp32",
            expected_source_sha256="0" * 64,
        )
