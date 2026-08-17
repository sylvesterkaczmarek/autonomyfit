from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from autonomyfit.artifacts import (
    ArtifactCacheError,
    ArtifactIntegrityError,
    ArtifactManager,
    ArtifactSecurityError,
    classify_candidate,
)
from autonomyfit.integrity import artifact_sha256
from autonomyfit.models import ModelProfile


def _model(*, license_status: str = "published") -> ModelProfile:
    return ModelProfile(
        id="demo-model",
        display_name="Demo model",
        family="demo",
        task="detection",
        params_m=1.0,
        source_id="hf-demo",
        source_url="https://huggingface.co/example/demo-model",
        runtimes=("onnx",),
        source_revision=None,
        license_spdx="Apache-2.0" if license_status == "published" else None,
        license_status=license_status,
    )


def _install_fake_hub(monkeypatch, tmp_path: Path, *, repo_sha: str, payload: bytes = b"onnx"):
    download = tmp_path / "download.onnx"
    download.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    class HfApi:
        def model_info(self, repo_id, revision, files_metadata):
            assert repo_id == "example/demo-model"
            assert files_metadata is True
            return SimpleNamespace(
                sha=repo_sha,
                siblings=[
                    SimpleNamespace(
                        rfilename="model.onnx",
                        size=len(payload),
                        lfs={"sha256": digest},
                    ),
                    SimpleNamespace(rfilename="modeling_demo.py", size=12, lfs=None),
                ],
                config={"auto_map": {"AutoModel": "modeling_demo.Demo"}},
            )

    module = types.ModuleType("huggingface_hub")
    module.HfApi = HfApi
    module.hf_hub_download = lambda **kwargs: str(download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return digest


def test_candidate_policy_refuses_code_pickle_and_serialized_engines():
    for name in ("model.py", "weights.pt", "weights.pkl", "model.engine"):
        assert classify_candidate(name).safe_static is False
    assert classify_candidate("model.onnx").safe_static is True
    assert classify_candidate("model.safetensors").safe_static is True


def test_local_artifact_hash_mismatch_is_fatal(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"abc")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        ArtifactManager(tmp_path / "cache").manage_local(
            _model(), path, expected_sha256="0" * 64
        )


def test_huggingface_revision_is_resolved_and_remote_code_is_never_required_for_metadata(monkeypatch, tmp_path):
    repo_sha = "a" * 40
    digest = _install_fake_hub(monkeypatch, tmp_path, repo_sha=repo_sha)
    manager = ArtifactManager(tmp_path / "cache")
    discovered = manager.discover_huggingface(_model(), revision="main")
    assert discovered["resolved_revision"] == repo_sha
    assert discovered["remote_code_required"] is True
    candidate = next(item for item in discovered["candidates"] if item["filename"] == "model.onnx")
    assert candidate["upstream_sha256"] == digest


def test_huggingface_requires_full_immutable_revision(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, tmp_path, repo_sha="abc123")
    with pytest.raises(ArtifactIntegrityError, match="full immutable commit"):
        ArtifactManager(tmp_path / "cache").discover_huggingface(_model(), revision="main")


def test_huggingface_download_verifies_upstream_hash_and_cache(monkeypatch, tmp_path):
    repo_sha = "b" * 40
    digest = _install_fake_hub(monkeypatch, tmp_path, repo_sha=repo_sha, payload=b"safe-onnx")
    manager = ArtifactManager(tmp_path / "cache")
    artifact = manager.acquire_huggingface(_model(), filename="model.onnx", revision="main")
    assert artifact.sha256 == digest
    assert artifact.resolved_revision == repo_sha
    assert artifact.remote_code_required is True
    assert artifact.license_spdx == "Apache-2.0"
    cached = manager.acquire_huggingface(
        _model(), filename="model.onnx", revision=repo_sha, offline=True
    )
    assert cached.sha256 == digest
    artifact.path.write_bytes(b"tampered")
    with pytest.raises(ArtifactCacheError, match="hash mismatch"):
        manager.cached_for_model(_model().id)


def test_restricted_licence_blocks_automatic_acquisition(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, tmp_path, repo_sha="c" * 40)
    manager = ArtifactManager(tmp_path / "cache")
    with pytest.raises(ArtifactSecurityError, match="licence status"):
        manager.acquire_huggingface(
            _model(license_status="restricted"), filename="model.onnx", revision="main"
        )
    artifact = manager.acquire_huggingface(
        _model(license_status="restricted"),
        filename="model.onnx",
        revision="main",
        allow_restricted_license=True,
    )
    assert artifact.license_status == "restricted"


def test_multifile_and_directory_artifact_identity_covers_all_members(tmp_path):
    xml = tmp_path / "model.xml"
    weights = tmp_path / "model.bin"
    xml.write_text("<net/>")
    weights.write_bytes(b"one")
    first = artifact_sha256(xml)
    weights.write_bytes(b"two")
    assert artifact_sha256(xml) != first

    package = tmp_path / "model.mlpackage"
    package.mkdir()
    (package / "Manifest.json").write_text("{}")
    nested = package / "Data"
    nested.mkdir()
    (nested / "weights.bin").write_bytes(b"x")
    first_package = artifact_sha256(package)
    (nested / "weights.bin").write_bytes(b"y")
    assert artifact_sha256(package) != first_package

def test_artifact_bundle_rejects_symbolic_links(tmp_path):
    package = tmp_path / "model.mlpackage"
    package.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (package / "Manifest.json").write_text("{}")
    (package / "weights.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        artifact_sha256(package)


def test_cache_record_cannot_escape_record_directory(tmp_path):
    manager = ArtifactManager(tmp_path / "cache")
    model = _model()
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"safe")
    managed = manager.manage_local(model, artifact)
    root = manager._record_root(model.id, managed.resolved_revision, managed.filename)
    root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"safe")
    record = managed.to_dict()
    record["path"] = "../../outside.onnx"
    record["cached"] = True
    (root / "artifact.json").write_text(json.dumps(record))
    with pytest.raises(ArtifactCacheError):
        manager._read_record(root)


def test_managed_artifact_identity_recheck_detects_mutation(tmp_path):
    from autonomyfit.artifacts import verify_artifact_identity

    path = tmp_path / "model.onnx"
    path.write_bytes(b"before")
    artifact = ArtifactManager(tmp_path / "cache").manage_local(_model(), path)
    path.write_bytes(b"after")
    with pytest.raises(ArtifactIntegrityError, match="identity changed"):
        verify_artifact_identity(artifact)
