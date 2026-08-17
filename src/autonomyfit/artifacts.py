from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from platformdirs import user_cache_path

from .integrity import artifact_sha256, artifact_size_bytes, sha256_file
from .models import ModelProfile

_SAFE_STATIC_SUFFIXES = {
    ".onnx",
    ".safetensors",
    ".json",
    ".txt",
    ".md",
    ".model",
    ".tflite",
    ".xml",
    ".mlmodel",
    ".mlpackage",
}
_UNSAFE_SUFFIXES = {
    ".py",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".bin",
    ".engine",
    ".plan",
    ".so",
    ".dll",
    ".dylib",
}
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


class ArtifactError(RuntimeError):
    """Base artifact-management error."""


class ArtifactSecurityError(ArtifactError):
    """Artifact handling would cross an unsafe execution boundary."""


class ArtifactIntegrityError(ArtifactError):
    """Artifact bytes do not match the expected identity."""


class ArtifactCacheError(ArtifactError):
    """Cached artifact metadata or bytes are corrupted."""


@dataclass(frozen=True)
class ArtifactCandidate:
    filename: str
    size_bytes: int | None
    safe_static: bool
    reason: str
    upstream_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManagedArtifact:
    model_id: str
    path: Path
    format: str
    sha256: str
    size_bytes: int
    source: str
    provenance_url: str | None
    requested_revision: str | None
    resolved_revision: str | None
    filename: str
    license_spdx: str | None
    license_status: str
    remote_code_required: bool
    trusted_for_execution: bool
    cached: bool
    acquired_at: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = str(self.path)
        return value


def verify_artifact_identity(artifact: ManagedArtifact) -> None:
    try:
        actual = artifact_sha256(artifact.path)
    except (OSError, ValueError) as exc:
        raise ArtifactIntegrityError(f"artifact identity could not be revalidated: {exc}") from exc
    if actual.casefold() != artifact.sha256.casefold():
        raise ArtifactIntegrityError(
            f"artifact identity changed: expected {artifact.sha256}, got {actual}"
        )


def artifact_format(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".safetensors":
        return "safetensors"
    if suffix in {".engine", ".plan"}:
        return "tensorrt-engine"
    if suffix == ".mlmodel":
        return "coreml-model"
    if suffix == ".mlpackage":
        return "coreml-package"
    if suffix == ".xml":
        return "openvino-ir"
    if suffix in {".pt", ".pth"}:
        return "pytorch"
    return suffix.lstrip(".") or "unknown"


def classify_candidate(
    filename: str,
    size_bytes: int | None = None,
    upstream_sha256: str | None = None,
) -> ArtifactCandidate:
    suffix = Path(filename).suffix.casefold()
    lower = filename.casefold()
    if suffix in _UNSAFE_SUFFIXES:
        return ArtifactCandidate(
            filename,
            size_bytes,
            False,
            "requires explicit trust because the format can contain executable or pickle/native state",
            upstream_sha256,
        )
    if lower.endswith((".onnx.data", ".onnx_data")):
        return ArtifactCandidate(filename, size_bytes, True, "ONNX external tensor data", upstream_sha256)
    if suffix in _SAFE_STATIC_SUFFIXES:
        return ArtifactCandidate(filename, size_bytes, True, "static data/model format", upstream_sha256)
    return ArtifactCandidate(filename, size_bytes, False, "format is not on the automatic safe list", upstream_sha256)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_id_from_model(model: ModelProfile) -> str | None:
    prefix = "https://huggingface.co/"
    if not model.source_url.startswith(prefix):
        return None
    path = model.source_url[len(prefix) :].strip("/")
    parts = path.split("/")
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def _safe_relpath(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactSecurityError(f"unsafe artifact path: {filename}")
    return path


def _require_licence_permission(model: ModelProfile, allow_restricted: bool) -> None:
    if model.license_status == "published":
        return
    if allow_restricted:
        return
    raise ArtifactSecurityError(
        "automatic artifact acquisition is blocked because the registry licence status is "
        f"{model.license_status!r}; inspect the upstream terms and re-run with "
        "--allow-restricted-license only if you have the required rights"
    )


class ArtifactManager:
    def __init__(self, cache_dir: Path | None = None) -> None:
        configured = os.environ.get("AUTONOMYFIT_ARTIFACT_DIR")
        self.cache_dir = (
            Path(configured).expanduser()
            if cache_dir is None and configured
            else cache_dir or user_cache_path("autonomyfit") / "artifacts"
        )

    def _record_root(self, model_id: str, revision: str | None, filename: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id):
            raise ArtifactSecurityError(f"unsafe model id for artifact cache: {model_id!r}")
        key = hashlib.sha256(
            f"{model_id}|{revision or 'unversioned'}|{filename}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / model_id / key

    def _metadata_path(self, root: Path) -> Path:
        return root / "artifact.json"

    def _write_record(self, artifact: ManagedArtifact) -> None:
        root = artifact.path.parent
        payload = artifact.to_dict()
        payload["path"] = artifact.path.name
        _atomic_write(
            self._metadata_path(root),
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )

    def _read_record(self, root: Path) -> ManagedArtifact | None:
        try:
            value = json.loads(self._metadata_path(root).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            relative = _safe_relpath(str(value["path"]))
            if len(relative.parts) != 1:
                raise ArtifactCacheError("cached artifact record must name one cache-local file")
            path = root / relative
            if path.is_symlink():
                raise ArtifactCacheError(f"cached artifact may not be a symbolic link: {path}")
            if not path.exists():
                raise ArtifactCacheError(f"cached artifact bytes are missing: {path}")
            if str(value["model_id"]) != root.parent.name:
                raise ArtifactCacheError("cached artifact model id does not match its cache directory")
            actual = artifact_sha256(path)
            expected = str(value["sha256"])
            if actual.casefold() != expected.casefold():
                raise ArtifactCacheError(
                    f"cached artifact hash mismatch for {path.name}: expected {expected}, got {actual}"
                )
            return ManagedArtifact(
                model_id=str(value["model_id"]),
                path=path,
                format=str(value["format"]),
                sha256=expected,
                size_bytes=int(value["size_bytes"]),
                source=str(value["source"]),
                provenance_url=value.get("provenance_url"),
                requested_revision=value.get("requested_revision"),
                resolved_revision=value.get("resolved_revision"),
                filename=str(value["filename"]),
                license_spdx=value.get("license_spdx"),
                license_status=str(value.get("license_status", "unknown")),
                remote_code_required=bool(value.get("remote_code_required", False)),
                trusted_for_execution=bool(value.get("trusted_for_execution", False)),
                cached=True,
                acquired_at=str(value["acquired_at"]),
            )
        except (KeyError, TypeError, ValueError, ArtifactSecurityError, OSError) as exc:
            raise ArtifactCacheError(f"invalid artifact cache record in {root}") from exc

    def manage_local(
        self,
        model: ModelProfile,
        path: Path,
        *,
        expected_sha256: str | None = None,
        trusted_for_execution: bool = False,
        revision: str | None = None,
    ) -> ManagedArtifact:
        path = path.expanduser().resolve()
        if not path.exists():
            raise ArtifactError(f"artifact does not exist: {path}")
        if path.is_dir() and path.suffix.casefold() != ".mlpackage":
            raise ArtifactError(f"unsupported artifact directory: {path}")
        try:
            digest = artifact_sha256(path)
        except (OSError, ValueError) as exc:
            raise ArtifactSecurityError(f"unsafe or unreadable artifact bundle: {exc}") from exc
        if expected_sha256 and digest.casefold() != expected_sha256.casefold():
            raise ArtifactIntegrityError(
                f"artifact SHA-256 mismatch: expected {expected_sha256}, got {digest}"
            )
        path.suffix.casefold()
        static = classify_candidate(path.name).safe_static
        trusted = trusted_for_execution or static
        return ManagedArtifact(
            model_id=model.id,
            path=path,
            format=artifact_format(path),
            sha256=digest,
            size_bytes=artifact_size_bytes(path),
            source="local",
            provenance_url=None,
            requested_revision=revision or model.source_revision,
            resolved_revision=revision or model.source_revision,
            filename=path.name,
            license_spdx=model.license_spdx,
            license_status=model.license_status,
            remote_code_required=False,
            trusted_for_execution=trusted,
            cached=False,
            acquired_at=_now(),
        )

    def discover_huggingface(
        self,
        model: ModelProfile,
        *,
        revision: str | None = None,
        offline: bool = False,
    ) -> dict[str, Any]:
        repo_id = _repo_id_from_model(model)
        if repo_id is None:
            raise ArtifactError(
                "automatic artifact discovery is currently available only for Hugging Face model sources"
            )
        if offline:
            cached = self.cached_for_model(model.id)
            return {
                "repo_id": repo_id,
                "requested_revision": revision or model.source_revision,
                "resolved_revision": None,
                "remote_code_required": False,
                "candidates": [],
                "cached": [item.to_dict() for item in cached],
                "offline": True,
            }
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ArtifactError(
                "Hugging Face artifact discovery requires: pip install 'autonomyfit[deployment]'"
            ) from exc
        requested = revision or model.source_revision or "main"
        try:
            info = HfApi().model_info(repo_id, revision=requested, files_metadata=True)
        except Exception as exc:  # huggingface_hub exposes several transport exceptions
            raise ArtifactError(f"could not inspect Hugging Face model {repo_id}: {exc}") from exc
        resolved = getattr(info, "sha", None)
        if not isinstance(resolved, str) or not _FULL_COMMIT.fullmatch(resolved):
            raise ArtifactIntegrityError(
                f"Hugging Face did not resolve {requested!r} to a full immutable commit SHA"
            )
        siblings = getattr(info, "siblings", None) or []
        candidates: list[ArtifactCandidate] = []
        has_python = False
        for sibling in siblings:
            name = getattr(sibling, "rfilename", None)
            if not isinstance(name, str):
                continue
            size = getattr(sibling, "size", None)
            lfs = getattr(sibling, "lfs", None)
            upstream_sha256 = (
                lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            )
            if not isinstance(upstream_sha256, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", upstream_sha256
            ):
                upstream_sha256 = None
            if name.casefold().endswith(".py"):
                has_python = True
            candidates.append(
                classify_candidate(
                    name, size if isinstance(size, int) else None, upstream_sha256
                )
            )
        config = getattr(info, "config", None)
        remote_code_required = has_python or bool(
            isinstance(config, dict) and config.get("auto_map")
        )
        return {
            "repo_id": repo_id,
            "requested_revision": requested,
            "resolved_revision": resolved,
            "remote_code_required": remote_code_required,
            "candidates": [item.to_dict() for item in candidates],
            "cached": [],
            "offline": False,
        }

    def acquire_huggingface(
        self,
        model: ModelProfile,
        *,
        filename: str | None = None,
        revision: str | None = None,
        expected_sha256: str | None = None,
        offline: bool = False,
        allow_restricted_license: bool = False,
    ) -> ManagedArtifact:
        _require_licence_permission(model, allow_restricted_license)
        if offline:
            matches = self.cached_for_model(model.id)
            if revision:
                matches = [item for item in matches if item.resolved_revision == revision]
            if filename:
                matches = [item for item in matches if item.filename == filename]
            if len(matches) != 1:
                raise ArtifactError(
                    "offline artifact acquisition needs exactly one matching verified cache entry"
                )
            artifact = matches[0]
            if expected_sha256 and artifact.sha256.casefold() != expected_sha256.casefold():
                raise ArtifactIntegrityError("cached artifact does not match requested SHA-256")
            return artifact

        discovery = self.discover_huggingface(model, revision=revision, offline=False)
        candidates = [ArtifactCandidate(**item) for item in discovery["candidates"]]
        if filename is None:
            safe_onnx = [item for item in candidates if item.safe_static and item.filename.casefold().endswith(".onnx")]
            safe_tensor = [
                item for item in candidates if item.safe_static and item.filename.casefold().endswith(".safetensors")
            ]
            pool = safe_onnx if len(safe_onnx) == 1 else safe_tensor if len(safe_tensor) == 1 else []
            if len(pool) != 1:
                safe_names = [item.filename for item in candidates if item.safe_static]
                preview = ", ".join(safe_names[:12]) or "none"
                raise ArtifactError(
                    "artifact selection is ambiguous; use --filename with one safe candidate. "
                    f"Safe candidates: {preview}"
                )
            filename = pool[0].filename
        _safe_relpath(filename)
        chosen = next((item for item in candidates if item.filename == filename), None)
        if chosen is None:
            raise ArtifactError(f"artifact file is not present at the pinned revision: {filename}")
        if not chosen.safe_static:
            raise ArtifactSecurityError(
                f"automatic download refused for {filename}: {chosen.reason}"
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ArtifactError(
                "Hugging Face artifact download requires: pip install 'autonomyfit[deployment]'"
            ) from exc
        repo_id = str(discovery["repo_id"])
        resolved = str(discovery["resolved_revision"])
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=resolved,
                    cache_dir=str(self.cache_dir / "hf"),
                    local_files_only=False,
                )
            )
        except Exception as exc:
            raise ArtifactError(f"Hugging Face download failed: {exc}") from exc
        digest = artifact_sha256(downloaded)
        required_digest = expected_sha256 or chosen.upstream_sha256
        if required_digest and digest.casefold() != required_digest.casefold():
            source = "requested" if expected_sha256 else "upstream LFS"
            raise ArtifactIntegrityError(
                f"artifact SHA-256 mismatch against {source} digest: expected {required_digest}, got {digest}"
            )
        root = self._record_root(model.id, resolved, filename)
        target = root / Path(filename).name
        _atomic_copy(downloaded, target)
        artifact = ManagedArtifact(
            model_id=model.id,
            path=target,
            format=artifact_format(target),
            sha256=digest,
            size_bytes=target.stat().st_size,
            source="huggingface",
            provenance_url=f"https://huggingface.co/{repo_id}/resolve/{resolved}/{filename}",
            requested_revision=str(discovery["requested_revision"]),
            resolved_revision=resolved,
            filename=filename,
            license_spdx=model.license_spdx,
            license_status=model.license_status,
            remote_code_required=bool(discovery["remote_code_required"]),
            trusted_for_execution=target.suffix.casefold() in {".onnx", ".safetensors"},
            cached=True,
            acquired_at=_now(),
        )
        self._write_record(artifact)
        return artifact

    def acquire_url(
        self,
        model: ModelProfile,
        *,
        url: str,
        expected_sha256: str,
        filename: str | None = None,
        revision: str | None = None,
        timeout: float = 30.0,
        allow_restricted_license: bool = False,
    ) -> ManagedArtifact:
        _require_licence_permission(model, allow_restricted_license)
        if not url.startswith("https://"):
            raise ArtifactSecurityError("remote artifact URLs must use HTTPS")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ArtifactIntegrityError("--sha256 must be a 64-character hexadecimal digest")
        name = filename or Path(urlparse(url).path).name
        if not name:
            name = "artifact"
        candidate = classify_candidate(name)
        if not candidate.safe_static:
            raise ArtifactSecurityError(f"automatic URL download refused: {candidate.reason}")
        request = urllib.request.Request(url, headers={"User-Agent": "autonomyfit-artifact-client/0.6"})
        fd, tmp_name = tempfile.mkstemp(prefix="autonomyfit-download-")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise ArtifactError(f"artifact download failed: {exc}") from exc
            digest = sha256_file(tmp)
            if digest.casefold() != expected_sha256.casefold():
                raise ArtifactIntegrityError(
                    f"artifact SHA-256 mismatch: expected {expected_sha256}, got {digest}"
                )
            root = self._record_root(model.id, revision, name)
            target = root / Path(name).name
            _atomic_copy(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        artifact = ManagedArtifact(
            model_id=model.id,
            path=target,
            format=artifact_format(target),
            sha256=expected_sha256.casefold(),
            size_bytes=target.stat().st_size,
            source="url",
            provenance_url=url,
            requested_revision=revision,
            resolved_revision=revision,
            filename=name,
            license_spdx=model.license_spdx,
            license_status=model.license_status,
            remote_code_required=False,
            trusted_for_execution=True,
            cached=True,
            acquired_at=_now(),
        )
        self._write_record(artifact)
        return artifact

    def cached_for_model(self, model_id: str) -> list[ManagedArtifact]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id):
            raise ArtifactSecurityError(f"unsafe model id for artifact cache: {model_id!r}")
        root = self.cache_dir / model_id
        if not root.exists():
            return []
        result: list[ManagedArtifact] = []
        for directory in sorted(
            path for path in root.iterdir() if path.is_dir() and not path.is_symlink()
        ):
            record = self._read_record(directory)
            if record is not None:
                result.append(record)
        return result
