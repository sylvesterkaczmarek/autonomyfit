from __future__ import annotations

import sys

import pytest

from autonomyfit.artifacts import (
    ArtifactIntegrityError,
    ArtifactManager,
    ArtifactSecurityError,
    verify_artifact_identity,
)
from autonomyfit.integrity import (
    artifact_members,
    artifact_sha256,
    artifact_size_bytes,
    sha256_file,
)
from autonomyfit.models import ModelProfile


def _model():
    return ModelProfile(
        id="local-model", display_name="Local model", family="test", task="classification",
        params_m=1.0, source_id="local", source_url="https://example.com/model",
        runtimes=("onnx",),
    )


def test_external_weights_change_prediction_and_artifact_identity(tmp_path, write_onnx):
    import numpy as np
    import onnxruntime as ort

    path = write_onnx(tmp_path / "model.onnx", external=True, location="data/weights.data")
    weight_path = tmp_path / "data/weights.data"
    artifact = ArtifactManager(tmp_path / "cache").manage_local(_model(), path)
    assert artifact_members(path) == (path, weight_path)
    assert artifact_size_bytes(path) == path.stat().st_size + weight_path.stat().st_size
    graph_hash = sha256_file(path)

    def predict():
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return session.run(None, {"input": np.array([1.0], dtype=np.float32)})[0].item()

    assert predict() == 2.0
    weight_path.write_bytes(np.array([5.0], dtype=np.float32).tobytes())
    assert predict() == 6.0
    assert sha256_file(path) == graph_hash
    with pytest.raises(ArtifactIntegrityError, match="identity changed"):
        verify_artifact_identity(artifact)


def test_constant_attribute_external_weights_are_included(tmp_path, write_onnx):
    import onnx
    from onnx import helper

    path = write_onnx(tmp_path / "model.onnx", external=True)
    model = onnx.load(path, load_external_data=False)
    weight = model.graph.initializer.pop()
    model.graph.node.insert(0, helper.make_node("Constant", [], ["weight"], value=weight))
    path.write_bytes(model.SerializeToString())
    before = artifact_sha256(path)
    (tmp_path / "weights.data").write_bytes(b"new!")
    assert artifact_sha256(path) != before


@pytest.mark.parametrize("location", ["../weights.data", "/tmp/weights.data", "C:\\weights.data", ""])
def test_unsafe_external_paths_are_rejected_before_reading(tmp_path, write_onnx, location):
    import onnx

    path = write_onnx(tmp_path / "model.onnx", external=True)
    model = onnx.load(path, load_external_data=False)
    for entry in model.graph.initializer[0].external_data:
        if entry.key == "location":
            entry.value = location
    path.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match="unsafe ONNX external tensor path"):
        artifact_sha256(path)


def test_missing_external_weights_are_rejected(tmp_path, write_onnx):
    path = write_onnx(tmp_path / "model.onnx", external=True)
    (tmp_path / "weights.data").unlink()
    with pytest.raises(ValueError, match="external tensor file is missing"):
        artifact_sha256(path)


@pytest.mark.parametrize("directory_link", [False, True])
def test_external_weights_cannot_follow_symlinks(tmp_path, write_onnx, directory_link):
    path = write_onnx(tmp_path / "model.onnx", external=True, location="data/weights.data")
    if directory_link:
        (tmp_path / "data").rename(tmp_path / "outside")
        (tmp_path / "data").symlink_to(tmp_path / "outside", target_is_directory=True)
    else:
        weights = tmp_path / "data/weights.data"
        weights.rename(tmp_path / "outside.data")
        weights.symlink_to(tmp_path / "outside.data")
    with pytest.raises(ValueError, match="symbolic links"):
        artifact_sha256(path)


def test_root_symlink_is_rejected_by_both_identity_functions(tmp_path):
    path = tmp_path / "real.engine"
    path.write_bytes(b"engine")
    link = tmp_path / "link.engine"
    link.symlink_to(path)
    for function in (artifact_members, artifact_sha256, artifact_size_bytes):
        with pytest.raises(ValueError, match="symbolic link"):
            function(link)
    with pytest.raises(ArtifactSecurityError, match="symbolic link"):
        # Managers must not resolve away this boundary before checking it.
        ArtifactManager(tmp_path / "cache").manage_local(_model(), link)


def test_missing_parser_cannot_silently_fall_back_to_graph_only_hash(tmp_path, write_onnx, monkeypatch):
    path = write_onnx(tmp_path / "model.onnx")
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(ValueError, match=r"install 'autonomyfit\[deployment\]'"):
        artifact_sha256(path)


def test_invalid_onnx_is_not_accepted_as_complete_artifact(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"invalid ONNX")
    with pytest.raises(ValueError, match="could not inspect ONNX artifact"):
        artifact_sha256(path)
