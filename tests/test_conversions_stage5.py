from __future__ import annotations

import pytest

from autonomyfit.conversions import (
    ConversionError,
    compare_onnx_openvino_outputs,
    convert_artifact,
    convert_to_openvino,
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


def test_generic_onnx_to_coreml_is_refused_without_trusted_source_contract(tmp_path, write_onnx):
    source = write_onnx(tmp_path / "model.onnx")
    with pytest.raises(ConversionError, match="intentionally not automated"):
        convert_artifact(source, "coreml", tmp_path / "out")


def test_identity_conversion_preserves_artifact_digest(tmp_path, write_onnx):
    source = write_onnx(tmp_path / "model.onnx")
    result = convert_artifact(source, "onnxruntime", tmp_path / "out", precision="fp32")
    assert result.source_sha256 == result.target_sha256
    assert result.target_path == source
    assert result.built_locally is False

def test_conversion_rejects_source_identity_mismatch(tmp_path, write_onnx):
    source = write_onnx(tmp_path / "model.onnx")
    with pytest.raises(ConversionError, match="changed before conversion"):
        convert_artifact(
            source,
            "onnxruntime",
            tmp_path / "out",
            precision="fp32",
            expected_source_sha256="0" * 64,
        )


def test_external_weight_mutation_during_conversion_is_rejected(tmp_path, write_onnx, monkeypatch):
    from autonomyfit.conversions import ConversionResult
    from autonomyfit.integrity import artifact_sha256

    source = write_onnx(tmp_path / "model.onnx", external=True)
    before = artifact_sha256(source)

    def convert(*args, **kwargs):
        (tmp_path / "weights.data").write_bytes(b"new!")
        return ConversionResult(
            source_path=source, source_sha256=before, source_format="onnx",
            target_path=source, target_sha256=before, target_format="onnx",
            target_runtime="onnxruntime", tool="test", tool_version=None,
            command=(), duration_s=0.0,
        )

    monkeypatch.setattr("autonomyfit.conversions._convert_artifact_unchecked", convert)
    with pytest.raises(ConversionError, match="source artifact changed during conversion"):
        convert_artifact(source, "onnxruntime", tmp_path / "out")


@pytest.mark.parametrize("precision,flag", [("fp16", "True"), ("fp32", "False"), ("artifact", "False")])
def test_openvino_cli_honours_weight_compression_request(tmp_path, write_onnx, monkeypatch, precision, flag):
    from pathlib import Path

    source = write_onnx(tmp_path / "model.onnx")
    monkeypatch.setattr("autonomyfit.conversions.shutil.which", lambda _: "/tools/ovc")
    monkeypatch.setattr("autonomyfit.conversions._tool_version", lambda _: "test")

    def run(command):
        assert f"--compress_to_fp16={flag}" in command
        target = Path(command[command.index("--output_model") + 1])
        target.write_bytes(b"<net/>")
        target.with_suffix(".bin").write_bytes(b"weights")
        return 0.1, ""

    monkeypatch.setattr("autonomyfit.conversions._run", run)
    result = convert_to_openvino(source, tmp_path / "out", precision=precision)
    assert f"--compress_to_fp16={flag}" in result.command
    assert len(result.companion_artifacts) == 1


@pytest.mark.parametrize("precision,compress", [("fp16", True), ("fp32", False), ("artifact", False)])
def test_openvino_python_matches_cli_weight_compression(tmp_path, write_onnx, monkeypatch, precision, compress):
    import sys
    from pathlib import Path
    from types import ModuleType

    source = write_onnx(tmp_path / "model.onnx")
    monkeypatch.setattr("autonomyfit.conversions.shutil.which", lambda _: None)
    module = ModuleType("openvino")
    module.convert_model = lambda _: object()

    def save(model, path, *, compress_to_fp16):
        assert compress_to_fp16 is compress
        Path(path).write_bytes(b"<net/>")
        Path(path).with_suffix(".bin").write_bytes(b"weights")

    module.save_model = save
    monkeypatch.setitem(sys.modules, "openvino", module)
    result = convert_to_openvino(source, tmp_path / "out", precision=precision)
    assert f"compress_to_fp16={compress}" in result.command


@pytest.mark.parametrize("conversion", [convert_to_openvino, convert_to_tensorrt])
def test_int8_is_refused_before_conversion_without_validated_quantization(tmp_path, write_onnx, monkeypatch, conversion):
    source = write_onnx(tmp_path / "model.onnx")
    monkeypatch.setattr("autonomyfit.conversions._run", lambda _: pytest.fail("converter must not run"))
    with pytest.raises(ConversionError, match="[Ii][Nn][Tt]8"):
        conversion(source, tmp_path / "out", precision="int8")
    assert not (tmp_path / "out").exists()


def _fake_equivalence_runtimes(monkeypatch, reference, converted):
    import sys
    from types import ModuleType, SimpleNamespace

    class Port:
        any_name = "input"

    port = Port()

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1], type="tensor(float)")]

        def run(self, *args):
            return [reference]

    class Compiled:
        inputs = (port,)
        outputs = (port,)

        def __call__(self, feeds):
            return {port: converted}

    ort = ModuleType("onnxruntime")
    ort.InferenceSession = Session
    ov = ModuleType("openvino")
    ov.Core = lambda: SimpleNamespace(compile_model=lambda *args: Compiled())
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "openvino", ov)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_matching_nonfinite_outputs_cannot_establish_equivalence(tmp_path, monkeypatch, value):
    import numpy as np

    output = np.array([value], dtype=np.float32)
    _fake_equivalence_runtimes(monkeypatch, output, output)
    result = compare_onnx_openvino_outputs(tmp_path / "source.onnx", tmp_path / "target.xml")
    assert result["status"] == "failed"
    assert "non-finite" in result["reason"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_equivalence_rejects_invalid_tolerances_before_execution(tmp_path, value):
    result = compare_onnx_openvino_outputs(tmp_path / "source.onnx", tmp_path / "target.xml", rtol=value)
    assert result["status"] == "failed"
    assert "tolerances" in result["reason"]


def test_finite_matching_outputs_still_establish_limited_equivalence(tmp_path, monkeypatch):
    import numpy as np

    _fake_equivalence_runtimes(monkeypatch, np.array([1.0]), np.array([1.0001]))
    result = compare_onnx_openvino_outputs(tmp_path / "source.onnx", tmp_path / "target.xml")
    assert result["status"] == "passed"
    assert result["max_abs_error"] == pytest.approx(0.0001)
