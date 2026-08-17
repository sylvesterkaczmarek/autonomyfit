from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(name: str) -> Path:
    return ROOT / name


def replace_once(name: str, old: str, new: str) -> None:
    path = _path(name)
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: expected exactly one replacement target, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_all(name: str, old: str, new: str, *, minimum: int = 1) -> None:
    path = _path(name)
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count < minimum:
        raise RuntimeError(f"{name}: expected at least {minimum} replacement targets, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def append_once(name: str, marker: str, addition: str) -> None:
    path = _path(name)
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + addition.rstrip() + "\n", encoding="utf-8")


# Patch release metadata.
replace_once("pyproject.toml", 'version = "0.6.0"', 'version = "0.6.1"')
replace_once("src/autonomyfit/__init__.py", '__version__ = "0.6.0"', '__version__ = "0.6.1"')
replace_once("CITATION.cff", 'date-released: "2026-08-16"', 'date-released: "2026-08-17"')
replace_once("CITATION.cff", "version: 0.6.0", "version: 0.6.1")
replace_all(".github/workflows/ci.yml", "0.6.0", "0.6.1", minimum=4)

# Benchmark report IDs are persistent filenames. Constrain them at the schema boundary.
replace_once(
    "src/autonomyfit/data/benchmark-report-v2.schema.json",
    '''    "benchmark_id": {
      "type": "string",
      "minLength": 8
    },''',
    '''    "benchmark_id": {
      "type": "string",
      "minLength": 8,
      "maxLength": 128,
      "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]*$"
    },''',
)

# Evidence import: reject future-dated evidence and make writes path-safe and atomic.
replace_once(
    "src/autonomyfit/evidence.py",
    "import os\nfrom dataclasses import asdict, dataclass, field\nfrom datetime import date, datetime, timezone",
    "import os\nimport tempfile\nfrom dataclasses import asdict, dataclass, field\nfrom datetime import date, datetime, timedelta, timezone",
)
replace_once(
    "src/autonomyfit/evidence.py",
    '''    created = datetime.fromisoformat(document["created_at"].replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise EvidenceSchemaError("benchmark report created_at must include a timezone")
    if document["quality"] != "local-measured":''',
    '''    created = datetime.fromisoformat(document["created_at"].replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise EvidenceSchemaError("benchmark report created_at must include a timezone")
    if created.astimezone(timezone.utc) > datetime.now(timezone.utc) + timedelta(minutes=10):
        raise EvidenceSchemaError("benchmark report created_at is implausibly far in the future")
    if document["quality"] != "local-measured":''',
)
replace_once(
    "src/autonomyfit/evidence.py",
    '''    target_dir = directory or local_benchmark_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{document['benchmark_id']}.json"
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\\n").encode()
    target.write_bytes(payload)
    return target''',
    '''    target_dir = (directory or local_benchmark_dir()).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = target_dir.resolve()
    benchmark_id = str(document["benchmark_id"])
    target = target_dir / f"{benchmark_id}.json"
    if target.parent != target_dir:
        raise EvidenceSchemaError("benchmark_id would escape the local evidence directory")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\\n").encode()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{benchmark_id}.", dir=target_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, target)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return target''',
)

# Local result validation: schema-check before trusting and reject future timestamps.
replace_once(
    "src/autonomyfit/local_results.py",
    "from datetime import datetime, timezone",
    "from datetime import datetime, timedelta, timezone",
)
replace_once(
    "src/autonomyfit/local_results.py",
    '''        created = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError
        age_days = (current - created.astimezone(timezone.utc)).days
        if age_days > max_age_days:
            reasons.append(f"local result is stale ({age_days} days > {max_age_days})")''',
    '''        created = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError
        created_utc = created.astimezone(timezone.utc)
        if created_utc > current + timedelta(minutes=10):
            reasons.append("local result timestamp is implausibly far in the future")
        age_days = (current - created_utc).days
        if age_days > max_age_days:
            reasons.append(f"local result is stale ({age_days} days > {max_age_days})")''',
)
replace_once(
    "src/autonomyfit/local_results.py",
    '''    if not isinstance(value, dict):
        return LocalResultStatus(
            path, None, None, False, ("report root is not a JSON object",), None, None, None, None
        )
    valid, reasons = local_report_compatibility(''',
    '''    if not isinstance(value, dict):
        return LocalResultStatus(
            path, None, None, False, ("report root is not a JSON object",), None, None, None, None
        )
    from .evidence import EvidenceError, validate_benchmark_report

    try:
        validate_benchmark_report(value)
    except EvidenceError as exc:
        return LocalResultStatus(
            path,
            value.get("benchmark_id"),
            (value.get("model") or {}).get("id"),
            False,
            (f"benchmark report validation failed: {exc}",),
            value.get("created_at"),
            (value.get("software") or {}).get("runtime"),
            (value.get("software") or {}).get("runtime_version"),
            (value.get("artifact") or {}).get("sha256"),
        )
    valid, reasons = local_report_compatibility(''',
)

# Artifact identity: symbolic links are not part of a self-contained deployment bundle.
replace_once(
    "src/autonomyfit/integrity.py",
    '''def artifact_members(path: Path) -> tuple[Path, ...]:
    """Return the byte-bearing members that define one deployment artifact identity."""
    path = path.expanduser().resolve()
    if path.is_dir():
        members = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
        if not members:
            raise ValueError(f"artifact directory is empty: {path}")
        return members
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {path}")
    if path.suffix.casefold() == ".xml":
        companion = path.with_suffix(".bin")
        if companion.is_file():
            return (path, companion)
    return (path,)''',
    '''def artifact_members(path: Path) -> tuple[Path, ...]:
    """Return the byte-bearing members that define one deployment artifact identity."""
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"artifact root may not be a symbolic link: {expanded}")
    path = expanded.resolve()
    if path.is_dir():
        entries = tuple(sorted(path.rglob("*")))
        symlinks = [item for item in entries if item.is_symlink()]
        if symlinks:
            raise ValueError(
                "artifact bundle may not contain symbolic links: "
                + ", ".join(str(item.relative_to(path)) for item in symlinks[:5])
            )
        members = tuple(item for item in entries if item.is_file())
        if not members:
            raise ValueError(f"artifact directory is empty: {path}")
        return members
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {path}")
    if path.suffix.casefold() == ".xml":
        companion = path.with_suffix(".bin")
        if companion.is_symlink():
            raise ValueError(f"OpenVINO companion may not be a symbolic link: {companion}")
        if companion.is_file():
            return (path, companion)
    return (path,)''',
)

# Managed artifact cache: bind model IDs and records to safe cache-local paths and expose an identity recheck.
replace_once(
    "src/autonomyfit/artifacts.py",
    '''    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = str(self.path)
        return value



def artifact_format(path: Path) -> str:''',
    '''    def to_dict(self) -> dict[str, Any]:
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


def artifact_format(path: Path) -> str:''',
)
replace_once(
    "src/autonomyfit/artifacts.py",
    '''    def _record_root(self, model_id: str, revision: str | None, filename: str) -> Path:
        key = hashlib.sha256(
            f"{model_id}|{revision or 'unversioned'}|{filename}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / model_id / key''',
    '''    def _record_root(self, model_id: str, revision: str | None, filename: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id):
            raise ArtifactSecurityError(f"unsafe model id for artifact cache: {model_id!r}")
        key = hashlib.sha256(
            f"{model_id}|{revision or 'unversioned'}|{filename}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / model_id / key''',
)
replace_once(
    "src/autonomyfit/artifacts.py",
    '''        try:
            path = root / str(value["path"])
            if not path.exists():
                raise ArtifactCacheError(f"cached artifact bytes are missing: {path}")
            actual = artifact_sha256(path)''',
    '''        try:
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
            actual = artifact_sha256(path)''',
)
replace_once(
    "src/autonomyfit/artifacts.py",
    '''        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactCacheError(f"invalid artifact cache record in {root}") from exc''',
    '''        except (KeyError, TypeError, ValueError, ArtifactSecurityError, OSError) as exc:
            raise ArtifactCacheError(f"invalid artifact cache record in {root}") from exc''',
)
replace_once(
    "src/autonomyfit/artifacts.py",
    '''        digest = artifact_sha256(path)
        if expected_sha256 and digest.casefold() != expected_sha256.casefold():''',
    '''        try:
            digest = artifact_sha256(path)
        except (OSError, ValueError) as exc:
            raise ArtifactSecurityError(f"unsafe or unreadable artifact bundle: {exc}") from exc
        if expected_sha256 and digest.casefold() != expected_sha256.casefold():''',
)
replace_once(
    "src/autonomyfit/artifacts.py",
    '''    def cached_for_model(self, model_id: str) -> list[ManagedArtifact]:
        root = self.cache_dir / model_id
        if not root.exists():
            return []
        result: list[ManagedArtifact] = []
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):''',
    '''    def cached_for_model(self, model_id: str) -> list[ManagedArtifact]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id):
            raise ArtifactSecurityError(f"unsafe model id for artifact cache: {model_id!r}")
        root = self.cache_dir / model_id
        if not root.exists():
            return []
        result: list[ManagedArtifact] = []
        for directory in sorted(
            path for path in root.iterdir() if path.is_dir() and not path.is_symlink()
        ):''',
)

# Registry cache: every cached document must still match the preserved accepted security state.
replace_once(
    "src/autonomyfit/registry.py",
    '''        cache_state = self._read_state(self.cache_state_path)
        expected_digest = cache_state.get("sha256")
        if not expected_digest or expected_digest != _sha256(registry_bytes):
            return None
        try:
            document = _read_json_bytes(registry_bytes, "cached registry")
            validate_registry_document(document)
        except RegistryError:
            return None
        return document, registry_bytes, bundle_bytes''',
    '''        digest = _sha256(registry_bytes)
        cache_state = self._read_state(self.cache_state_path)
        expected_digest = cache_state.get("sha256")
        if not expected_digest or expected_digest != digest:
            return None
        try:
            document = _read_json_bytes(registry_bytes, "cached registry")
            validate_registry_document(document)
        except RegistryError:
            return None
        security_state = self._read_state(self.security_state_path)
        highest = security_state.get("highest_seen_version")
        trusted_digest = security_state.get("sha256")
        version = int(document["registry"]["registry_version"])
        if not isinstance(highest, int) or not isinstance(trusted_digest, str):
            return None
        if version != highest or digest != trusted_digest:
            return None
        return document, registry_bytes, bundle_bytes''',
)

# Discovery provenance: require immutable Hub SHAs and prevent source collisions from contaminating curated records.
replace_once(
    "src/autonomyfit/discovery.py",
    '''        source_url = f"https://huggingface.co/{model_id}"
        runtimes = _runtime_tags(payload, mapped_task)
        return DiscoveryCandidate(''',
    '''        source_url = f"https://huggingface.co/{model_id}"
        runtimes = _runtime_tags(payload, mapped_task)
        revision = str(payload.get("sha")) if payload.get("sha") else None
        if revision and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            warnings.append("Hub revision is not a full immutable commit SHA")
            revision = None
        return DiscoveryCandidate(''',
)
replace_once(
    "src/autonomyfit/discovery.py",
    '''            revision=str(payload.get("sha")) if payload.get("sha") else None,''',
    '''            revision=revision,''',
)
replace_once(
    "src/autonomyfit/discovery.py",
    '''def _candidate_model_id(
    candidate: DiscoveryCandidate,
    existing_by_url: dict[str, str],
    used_ids: set[str],
) -> str:
    existing = existing_by_url.get(candidate.source_url.rstrip("/").casefold())
    if existing:
        return existing
    base = _slug(candidate.display_name)
    if not base:
        base = _slug(candidate.upstream_id)
    if base in used_ids:
        return base
    candidate_id = base
    suffix = 2
    original = candidate_id
    while candidate_id in used_ids:
        candidate_id = f"{original}-{suffix}"
        suffix += 1
    return candidate_id''',
    '''def _candidate_model_id(
    candidate: DiscoveryCandidate,
    existing_by_url: dict[str, str],
    existing_by_id: dict[str, dict[str, Any]],
    used_ids: set[str],
) -> str:
    existing = existing_by_url.get(candidate.source_url.rstrip("/").casefold())
    if existing:
        return existing
    base = _slug(candidate.display_name) or _slug(candidate.upstream_id)
    if base not in used_ids:
        return base

    occupied = existing_by_id.get(base)
    family = occupied.get("family") if isinstance(occupied, dict) else None
    same_family = bool(
        isinstance(family, dict)
        and _slug(str(family.get("name") or "")) == _slug(candidate.family)
        and _slug(str(family.get("variant") or "")) == _slug(candidate.variant or "")
    )
    upstream_slug = _slug(candidate.upstream_id.rsplit("/", 1)[-1])
    if occupied is not None and same_family and upstream_slug == base:
        return base

    candidate_id = base
    suffix = 2
    while candidate_id in used_ids:
        candidate_id = f"{base}-{suffix}"
        suffix += 1
    return candidate_id''',
)
replace_once(
    "src/autonomyfit/discovery.py",
    '''        model_id = _candidate_model_id(candidate, existing_by_url, used_ids)''',
    '''        model_id = _candidate_model_id(
            candidate, existing_by_url, existing_by_id, used_ids
        )''',
)
replace_once(
    "src/autonomyfit/discovery.py",
    '''    for runtime in discovered["compatibility"]["runtimes"]:
        if runtime not in output["compatibility"]["runtimes"]:
            output["compatibility"]["runtimes"].append(runtime)
    output["compatibility"]["runtimes"] = sorted(output["compatibility"]["runtimes"])

    if same_source:''',
    '''    if same_source:
        for runtime in discovered["compatibility"]["runtimes"]:
            if runtime not in output["compatibility"]["runtimes"]:
                output["compatibility"]["runtimes"].append(runtime)
        output["compatibility"]["runtimes"] = sorted(
            output["compatibility"]["runtimes"]
        )

    if same_source:''',
)

# Benchmark execution: bind a run to the admitted artifact and detect mutation during execution.
replace_once(
    "src/autonomyfit/backends.py",
    '''from .benchmark import (
    MemorySampler,''',
    '''from .benchmark import (
    MemorySampler,''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''from .models import HardwareProfile''',
    '''from .integrity import artifact_sha256
from .models import HardwareProfile''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''    command: str | None = None
    trusted_artifact: bool = False''',
    '''    command: str | None = None
    trusted_artifact: bool = False
    expected_sha256: str | None = None''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''def run_benchmark(request: BenchmarkRequest, backend_name: str | None = None) -> dict[str, Any]:
    backend = get_backend(backend_name or infer_backend(request.model_path))
    return backend.benchmark(request)''',
    '''def run_benchmark(request: BenchmarkRequest, backend_name: str | None = None) -> dict[str, Any]:
    try:
        before = artifact_sha256(request.model_path)
    except (OSError, ValueError) as exc:
        raise BackendError(f"artifact identity could not be established before benchmark: {exc}") from exc
    if request.expected_sha256 and before.casefold() != request.expected_sha256.casefold():
        raise BackendError(
            f"artifact identity changed before benchmark: expected {request.expected_sha256}, got {before}"
        )
    backend = get_backend(backend_name or infer_backend(request.model_path))
    report = backend.benchmark(request)
    try:
        after = artifact_sha256(request.model_path)
    except (OSError, ValueError) as exc:
        raise BackendError(f"artifact identity could not be revalidated after benchmark: {exc}") from exc
    if after.casefold() != before.casefold():
        raise BackendError(
            f"artifact changed during benchmark: expected {before}, got {after}"
        )
    reported = (report.get("artifact") or {}).get("sha256")
    if not isinstance(reported, str) or reported.casefold() != before.casefold():
        raise BackendError("benchmark report artifact identity does not match the measured input")
    return report''',
)

# Conversion execution: verify the source identity before and after vendor/runtime tooling executes.
replace_once(
    "src/autonomyfit/conversions.py",
    '''def convert_artifact(
    source: Path,''',
    '''def _convert_artifact_unchecked(
    source: Path,''',
)
replace_once(
    "src/autonomyfit/conversions.py",
    '''    raise ConversionError(f"no safe conversion path from {fmt} to {target_runtime}")


def compare_onnx_openvino_outputs(''',
    '''    raise ConversionError(f"no safe conversion path from {fmt} to {target_runtime}")


def convert_artifact(
    source: Path,
    target_runtime: str,
    output_dir: Path,
    *,
    precision: str = "fp16",
    input_shape: list[int] | None = None,
    input_shapes: dict[str, list[int]] | None = None,
    trust_source: bool = False,
    expected_source_sha256: str | None = None,
) -> ConversionResult:
    try:
        before = artifact_sha256(source)
    except (OSError, ValueError) as exc:
        raise ConversionError(f"source artifact identity could not be established: {exc}") from exc
    if expected_source_sha256 and before.casefold() != expected_source_sha256.casefold():
        raise ConversionError(
            f"source artifact identity changed before conversion: expected {expected_source_sha256}, got {before}"
        )
    result = _convert_artifact_unchecked(
        source,
        target_runtime,
        output_dir,
        precision=precision,
        input_shape=input_shape,
        input_shapes=input_shapes,
        trust_source=trust_source,
    )
    try:
        after = artifact_sha256(source)
    except (OSError, ValueError) as exc:
        raise ConversionError(f"source artifact identity could not be revalidated: {exc}") from exc
    if after.casefold() != before.casefold() or result.source_sha256.casefold() != before.casefold():
        raise ConversionError(
            f"source artifact changed during conversion: expected {before}, got {after}"
        )
    return result


def compare_onnx_openvino_outputs(''',
)

# Stale signed-registry data remains usable offline, but cannot retain high recommendation confidence.
replace_once(
    "src/autonomyfit/scoring.py",
    '''    target_quantities: int,
) -> ConfidenceBreakdown:''',
    '''    target_quantities: int,
    registry_stale: bool,
) -> ConfidenceBreakdown:''',
)
replace_once(
    "src/autonomyfit/scoring.py",
    '''    score = 100.0 * sum(components) / len(components)
    if unresolved:
        score = min(score, 55.0)
    score = round(max(0.0, min(100.0, score)), 1)
    return ConfidenceBreakdown(''',
    '''    score = 100.0 * sum(components) / len(components)
    confidence_unresolved = list(unresolved)
    if registry_stale:
        confidence_unresolved.append("registry freshness")
    if confidence_unresolved:
        score = min(score, 55.0)
    score = round(max(0.0, min(100.0, score)), 1)
    return ConfidenceBreakdown(''',
)
replace_once(
    "src/autonomyfit/scoring.py",
    '''        unresolved_constraints=tuple(unresolved),''',
    '''        unresolved_constraints=tuple(dict.fromkeys(confidence_unresolved)),''',
)
replace_once(
    "src/autonomyfit/scoring.py",
    '''        performance_unknown = False
        power_unknown = False

        if not runtime_compatible:''',
    '''        performance_unknown = False
        power_unknown = False

        if loaded.provenance.stale:
            reasons.append(
                "registry data is stale; confidence is capped until a fresh signed registry is available"
            )
            unknowns.append("current registry freshness")

        if not runtime_compatible:''',
)
replace_once(
    "src/autonomyfit/scoring.py",
    '''            known_quantities=max(0, known_count),
            target_quantities=max(1, target_count),
        )''',
    '''            known_quantities=max(0, known_count),
            target_quantities=max(1, target_count),
            registry_stale=loaded.provenance.stale,
        )''',
)

# Deployment workflow: revalidate admitted artifacts, bind conversion/benchmark identities, and rerank assess results by exact artifacts.
replace_once(
    "src/autonomyfit/deployment.py",
    '''    ArtifactManager,
    ManagedArtifact,
)''',
    '''    ArtifactManager,
    ManagedArtifact,
    verify_artifact_identity,
)''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''from .reporting import recommendation_dict
from .scoring import choose_precision, choose_runtime, recommend_models''',
    '''from .ranking import rank_recommendations
from .reporting import recommendation_dict
from .scoring import choose_precision, choose_runtime, recommend_models''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''    except ArtifactError as exc:
        raise DeploymentValidationError(str(exc)) from exc

    checks, artifact_warnings = structural_checks(''',
    '''    except ArtifactError as exc:
        raise DeploymentValidationError(str(exc)) from exc

    try:
        verify_artifact_identity(managed)
    except ArtifactError as exc:
        raise DeploymentValidationError(str(exc)) from exc

    checks, artifact_warnings = structural_checks(''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''                input_shapes=({"input": options.shape} if options.shape else None),
                trust_source=options.trust_artifact,
            )''',
    '''                input_shapes=({"input": options.shape} if options.shape else None),
                trust_source=options.trust_artifact,
                expected_source_sha256=managed.sha256,
            )''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''                command=command,
                trusted_artifact=final_artifact.trusted_for_execution,
            )''',
    '''                command=command,
                trusted_artifact=final_artifact.trusted_for_execution,
                expected_sha256=final_artifact.sha256,
            )''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''    ranked = recommend_models(
        hardware,
        Constraints(
            task=next(iter(tasks)),
            runtime=runtime,
            precision=precision,
            include_experimental=True,
        ),
        offline=offline,
    )
    chosen = [item for item in ranked if item.model.id in set(model_ids)]
    return {
        "hardware": hardware.to_dict(),
        "reports": reports,
        "reordered_recommendations": [recommendation_dict(item) for item in chosen],
    }''',
    '''    report_by_model = {str(item["model"]["id"]): item for item in reports}
    chosen = []
    for model in selected:
        report = report_by_model[model.id]
        artifact = report.get("artifact") or {}
        exact = recommend_models(
            hardware,
            Constraints(
                task=next(iter(tasks)),
                model_id=model.id,
                model_revision=(report.get("model") or {}).get("revision"),
                artifact_sha256=artifact.get("sha256"),
                runtime=runtime,
                precision=precision,
                include_experimental=True,
            ),
            offline=offline,
        )
        if exact:
            chosen.append(exact[0])
    chosen = rank_recommendations(chosen, "balanced")
    return {
        "hardware": hardware.to_dict(),
        "reports": reports,
        "reordered_recommendations": [recommendation_dict(item) for item in chosen],
    }''',
)

# Documentation corrections and security boundaries.
replace_once(
    "docs/discovery.md",
    "| `HuggingFaceAdapter` | Hugging Face Hub model APIs | Broad discovery for supported tasks. |",
    "| `HuggingFaceAdapter` | Hugging Face Hub model APIs | Broad discovery for object detection and compact VLMs. |",
)
replace_once(
    "docs/discovery.md",
    '''Only supported task families are considered for registry promotion.

Current automatic promotion caps are intentionally edge-oriented:''',
    '''The selection engine supports ten task categories, but scheduled Hub promotion currently covers object detection and compact VLMs only. The other task categories use curated registry entries until equally conservative provider-specific discovery and normalization rules exist.

Current automatic promotion caps are intentionally edge-oriented:''',
)
replace_once(
    "README.md",
    '''## Current limitations

- Exact deployment evidence is still sparse across the full hardware x runtime x model matrix.''',
    '''## Current limitations

- Scheduled automatic model promotion currently covers object detection and compact VLMs; the other supported task categories are curated rather than continuously discovered.
- Exact deployment evidence is still sparse across the full hardware x runtime x model matrix.''',
)
replace_once(
    "docs/registry.md",
    '''A successful signature is necessary but not sufficient. The client also stores the highest
accepted registry version and digest. It rejects a lower version and rejects different content
that reuses the same version. The signed document contains generation and expiry times so a
stale/frozen registry is detectable.''',
    '''A successful signature is necessary but not sufficient. The client also stores the highest
accepted registry version and digest. Every cached registry load is rebound to that preserved
security state, so changing both cached bytes and mutable cache metadata cannot bypass the last
accepted version/digest. It rejects a lower version and rejects different content that reuses
the same version. The signed document contains generation and expiry times so a stale/frozen
registry is detectable. Stale data can remain available offline, but recommendation confidence
is capped until a fresh signed registry is available.''',
)
replace_once(
    "docs/benchmarking.md",
    '''Imported reports are schema validated. Corrupt or malformed reports are not added to the local evidence store.''',
    '''Imported reports are schema validated, benchmark IDs are constrained to cache-safe filenames, and writes are atomic. Corrupt, path-escaping or malformed reports are not added to the local evidence store.

The artifact digest is established before native benchmarking and checked again after execution. If the artifact changes during the run, AutonomyFit discards the benchmark instead of attaching the measurements to the wrong bytes.''',
)
replace_once(
    "docs/deployment.md",
    '''OpenVINO IR is a multi-file identity when a sibling `.bin` exists. Core ML `.mlpackage` artifacts are directory identities. These use a deterministic manifest digest over relative member names and each member's byte SHA-256. Modifying any member changes the deployment artifact identity.''',
    '''OpenVINO IR is a multi-file identity when a sibling `.bin` exists. Core ML `.mlpackage` artifacts are directory identities. These use a deterministic manifest digest over relative member names and each member's byte SHA-256. Symbolic links are rejected so a bundle cannot silently pull bytes from outside its identity boundary. Modifying any member changes the deployment artifact identity.

Before conversion or benchmarking, the admitted digest is rechecked. Conversion and benchmarking also compare the source digest before and after execution and fail rather than recording evidence if the input identity changes.''',
)
replace_once(
    "docs/security.md",
    '''- cache digest revalidation on every cache load
- path-traversal checks for ONNX external tensor locations''',
    '''- cache digest revalidation on every cache load
- cached registry bytes rebound to the preserved accepted version/digest state
- path-traversal checks for ONNX external tensor locations
- path-safe atomic local benchmark imports
- symbolic-link rejection for multi-file artifact identities
- pre/post identity checks around conversion and benchmarking''',
)

# Tests for the audit findings.
append_once(
    "tests/test_evidence.py",
    "def test_benchmark_import_rejects_path_escaping_id",
    '''def test_benchmark_import_rejects_path_escaping_id(tmp_path):
    report = _report()
    report["benchmark_id"] = "../../outside-report"
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report))
    with pytest.raises(EvidenceSchemaError):
        import_benchmark_report(source, tmp_path / "store")
    assert not (tmp_path / "outside-report.json").exists()


def test_future_dated_benchmark_report_is_rejected():
    report = _report()
    report["created_at"] = "2099-01-01T00:00:00Z"
    with pytest.raises(EvidenceSchemaError, match="future"):
        validate_benchmark_report(report)''',
)
append_once(
    "tests/test_local_results_stage5.py",
    "def test_future_dated_local_result_is_invalid",
    '''def test_future_dated_local_result_is_invalid():
    hardware = _hardware()
    future = _report(hardware, created="2099-01-01T00:00:00Z")
    valid, reasons = local_report_compatibility(
        future, hardware, now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("future" in reason for reason in reasons)''',
)
append_once(
    "tests/test_registry.py",
    "def test_cache_cannot_override_preserved_security_state",
    '''def test_cache_cannot_override_preserved_security_state(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document(version=7)))
    client.update()

    tampered = _document(version=8)
    tampered_bytes = _bytes(tampered)
    client.registry_path.write_bytes(tampered_bytes)
    state = json.loads(client.cache_state_path.read_text())
    import hashlib
    state["sha256"] = hashlib.sha256(tampered_bytes).hexdigest()
    state["registry_version"] = 8
    client.cache_state_path.write_text(json.dumps(state))

    snapshot = client.load(offline=True)
    assert snapshot.provenance.source == "bundled-fallback"''',
)
append_once(
    "tests/test_artifacts_stage5.py",
    "def test_artifact_bundle_rejects_symbolic_links",
    '''def test_artifact_bundle_rejects_symbolic_links(tmp_path):
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
        verify_artifact_identity(artifact)''',
)
append_once(
    "tests/test_discovery.py",
    "def test_huggingface_short_revision_is_not_source_verified",
    '''def test_huggingface_short_revision_is_not_source_verified():
    detail = _hf_detail(sha="abc123")
    adapter = HuggingFaceAdapter(
        transport=_hf_transport(detail),
        trusted_publishers={"HuggingFaceTB"},
    )
    model = adapter.discover(NOW)[0]
    assert model.revision is None
    assert model.lifecycle() == "NORMALIZED"
    assert candidate_to_registry_model(
        model, model_id="short-revision", checked_at="2026-08-16T10:00:00Z"
    ) is None


def test_slug_collision_from_unrelated_source_gets_unique_id():
    registry = _registry()
    existing = next(item for item in registry["models"] if item["id"] == "yolo26n")
    original_runtimes = list(existing["compatibility"]["runtimes"])
    collision = _candidate(
        upstream_id="Vendor/Unrelated-Model",
        source_url="https://huggingface.co/Vendor/Unrelated-Model",
        display_name="YOLO26n",
        family="Unrelated",
        variant="other",
        runtimes=("transformers",),
    )
    result = apply_discovery(registry, [collision], now=NOW)
    assert any(
        item["id"].startswith("yolo26n-")
        and item["upstream"]["source_url"] == collision.source_url
        for item in result.registry["models"]
    )
    preserved = next(item for item in result.registry["models"] if item["id"] == "yolo26n")
    assert preserved["compatibility"]["runtimes"] == original_runtimes''',
)
# Existing Hub tests should model the immutable SHA that the production API supplies.
replace_once(
    "tests/test_discovery.py",
    '''    sha="abc123",''',
    '''    sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",''',
)
replace_once(
    "tests/test_discovery.py",
    '''    assert model.revision == "abc123"''',
    '''    assert model.revision == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"''',
)
replace_once(
    "tests/test_discovery.py",
    '''    mirror = _candidate(
        upstream_id="ultralytics/yolo26n",
        source_url="https://huggingface.co/ultralytics/yolo26n",
        publisher="ultralytics",
        display_name="YOLO26n",
        family="YOLO26",
        variant="n",
        params_m=2.4,
    )''',
    '''    mirror = _candidate(
        upstream_id="ultralytics/yolo26n",
        source_url="https://huggingface.co/ultralytics/yolo26n",
        publisher="ultralytics",
        display_name="YOLO26n",
        family="YOLO26",
        variant="n",
        params_m=2.4,
        runtimes=("transformers",),
    )''',
)
replace_once(
    "tests/test_discovery.py",
    '''    assert matches[0]["upstream"]["source_url"] == original["upstream"]["source_url"]''',
    '''    assert matches[0]["upstream"]["source_url"] == original["upstream"]["source_url"]
    assert matches[0]["compatibility"]["runtimes"] == original["compatibility"]["runtimes"]''',
)
append_once(
    "tests/test_conversions_stage5.py",
    "def test_conversion_rejects_source_identity_mismatch",
    '''def test_conversion_rejects_source_identity_mismatch(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    with pytest.raises(ConversionError, match="changed before conversion"):
        convert_artifact(
            source,
            "onnxruntime",
            tmp_path / "out",
            precision="fp32",
            expected_source_sha256="0" * 64,
        )''',
)
append_once(
    "tests/test_backends.py",
    "def test_run_benchmark_rejects_artifact_mutation",
    '''def test_run_benchmark_rejects_artifact_mutation(monkeypatch, tmp_path):
    from autonomyfit.backends import run_benchmark
    from autonomyfit.integrity import artifact_sha256

    path = tmp_path / "model.onnx"
    path.write_bytes(b"before")
    before = artifact_sha256(path)

    class FakeBackend:
        def benchmark(self, request):
            report = {"artifact": {"sha256": before}}
            path.write_bytes(b"after")
            return report

    monkeypatch.setattr("autonomyfit.backends.get_backend", lambda name: FakeBackend())
    request = BenchmarkRequest(
        model_path=path,
        model_id="model",
        model_revision="revision",
        hardware=HARDWARE,
        expected_sha256=before,
    )
    with pytest.raises(BackendError, match="changed during benchmark"):
        run_benchmark(request, "fake")''',
)
append_once(
    "tests/test_confidence_stage4.py",
    "def test_stale_registry_caps_recommendation_confidence",
    '''def test_stale_registry_caps_recommendation_confidence(monkeypatch):
    from autonomyfit.catalog import LoadedCatalog, load_model_catalog
    from autonomyfit.models import RegistryProvenance

    loaded = load_model_catalog(offline=True)
    model = next(item for item in loaded.models if item.id == "yolo26n")
    stale = LoadedCatalog(
        models=(model,),
        provenance=RegistryProvenance(source="cache", stale=True, signature_verified=True),
    )
    monkeypatch.setattr("autonomyfit.scoring.load_model_catalog", lambda *args, **kwargs: stale)
    item = recommend_models(
        _hardware(), _constraints(), offline=True, evidence_store=_store("2026-08-16")
    )[0]
    assert item.confidence is not None
    assert item.confidence.score <= 55
    assert "registry freshness" in item.confidence.unresolved_constraints
    assert "current registry freshness" in item.unknowns''',
)
append_once(
    "tests/test_deployment_stage5.py",
    "def test_candidate_assessment_reranks_exact_supplied_artifacts",
    '''def test_candidate_assessment_reranks_exact_supplied_artifacts(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from autonomyfit.deployment import assess_candidates

    first = replace(
        _model(), id="first", source_url="https://example.com/first",
        runtimes=("onnx",), supported_precisions=("fp32",),
    )
    second = replace(
        _model(), id="second", source_url="https://example.com/second",
        runtimes=("onnx",), supported_precisions=("fp32",),
    )
    hashes = {"first": "1" * 64, "second": "2" * 64}

    def fake_validate(options):
        return {
            "model": {"id": options.model_id, "revision": f"rev-{options.model_id}"},
            "artifact": {"sha256": hashes[options.model_id]},
        }

    calls = []

    def fake_recommend(hardware, constraints, **kwargs):
        calls.append(
            (constraints.model_id, constraints.model_revision, constraints.artifact_sha256)
        )
        return [SimpleNamespace(model=SimpleNamespace(id=constraints.model_id))]

    monkeypatch.setattr("autonomyfit.deployment.validate_deployment", fake_validate)
    monkeypatch.setattr("autonomyfit.deployment._resolve_hardware", lambda profile: _fake_hardware())
    monkeypatch.setattr(
        "autonomyfit.deployment.load_model_catalog",
        lambda **kwargs: SimpleNamespace(models=(first, second)),
    )
    monkeypatch.setattr("autonomyfit.deployment.recommend_models", fake_recommend)
    monkeypatch.setattr("autonomyfit.deployment.rank_recommendations", lambda items, objective: items)
    monkeypatch.setattr(
        "autonomyfit.deployment.recommendation_dict",
        lambda item: {"model_id": item.model.id},
    )

    result = assess_candidates(
        ["first", "second"],
        {"first": tmp_path / "first.onnx", "second": tmp_path / "second.onnx"},
        runtime="onnx",
        precision="fp32",
        offline=True,
    )
    assert calls == [
        ("first", "rev-first", hashes["first"]),
        ("second", "rev-second", hashes["second"]),
    ]
    assert [item["model_id"] for item in result["reordered_recommendations"]] == [
        "first", "second"
    ]''',
)

# Pin every third-party GitHub Action to the immutable commit currently referenced by its intended tag/branch.
workflow_pins = {
    "actions/checkout@v5": "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09 # v5",
    "actions/setup-python@v6": "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6",
    "actions/upload-artifact@v4": "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4",
    "actions/download-artifact@v4": "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093 # v4",
    "sigstore/gh-action-sigstore-python@v3.2.0": "sigstore/gh-action-sigstore-python@a5caf349bc536fbef3668a10ed7f5cd309a4b53d # v3.2.0",
    "pypa/gh-action-pypi-publish@release/v1": "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33 # release/v1",
}
for workflow in sorted((ROOT / ".github/workflows").glob("*.yml")):
    text = workflow.read_text(encoding="utf-8")
    for mutable, pinned in workflow_pins.items():
        text = text.replace(mutable, pinned)
    workflow.write_text(text, encoding="utf-8")

workflow_test = '''from __future__ import annotations

import re
from pathlib import Path


def test_external_github_actions_are_pinned_to_full_commit_sha():
    workflows = Path(".github/workflows")
    failures = []
    for path in sorted(workflows.glob("*.yml")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith("uses:") and not stripped.startswith("- uses:"):
                continue
            value = stripped.split("uses:", 1)[1].strip().split(" #", 1)[0].strip().strip("\\\"'")
            if value.startswith("./"):
                continue
            if not re.fullmatch(r"[^@\\s]+@[0-9a-f]{40}", value):
                failures.append(f"{path}:{line_number}: {value}")
    assert not failures, "mutable or unpinned workflow actions: " + "; ".join(failures)
'''
_path("tests/test_workflow_security.py").write_text(workflow_test, encoding="utf-8")

print("AutonomyFit hostile-audit hardening patch applied")
