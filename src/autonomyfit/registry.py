from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from platformdirs import user_cache_path

from .models import AccuracyMetric, ModelProfile, RegistryProvenance

REGISTRY_SCHEMA_VERSION = 2
DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/sylvesterkaczmarek/autonomyfit/"
    "main/registry/published/registry-v2.json"
)
DEFAULT_BUNDLE_URL = f"{DEFAULT_REGISTRY_URL}.sigstore.json"
EXPECTED_CERT_IDENTITY = (
    "https://github.com/sylvesterkaczmarek/autonomyfit/"
    ".github/workflows/registry-publish.yml@refs/heads/main"
)
EXPECTED_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
DEFAULT_REFRESH_SECONDS = 24 * 60 * 60
MAX_CLOCK_SKEW = timedelta(minutes=10)


class RegistryError(RuntimeError):
    """Base registry error."""


class RegistrySchemaError(RegistryError):
    """Registry document does not conform to the supported schema."""


class RegistryTrustError(RegistryError):
    """Registry signature or trust verification failed."""


class RegistryRollbackError(RegistryTrustError):
    """Registry version or digest would roll back previously trusted state."""


class RegistryUnavailableError(RegistryError):
    """No usable registry could be loaded."""


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes = b""
    etag: str | None = None


@dataclass(frozen=True)
class RegistrySnapshot:
    document: dict[str, Any]
    models: tuple[ModelProfile, ...]
    provenance: RegistryProvenance


Fetcher = Callable[[str, str | None, float], HttpResult]
Verifier = Callable[[bytes, bytes], None]
NowFn = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RegistrySchemaError(f"invalid registry timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise RegistrySchemaError(f"registry timestamp lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _read_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistrySchemaError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RegistrySchemaError(f"{label} must be a JSON object")
    return value


def _load_schema() -> dict[str, Any]:
    resource = files("autonomyfit.data").joinpath("registry-v2.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_registry_document(document: dict[str, Any]) -> None:
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise RegistrySchemaError(
            f"unsupported registry schema {document.get('schema_version')!r}; "
            f"expected {REGISTRY_SCHEMA_VERSION}"
        )
    validator = Draft202012Validator(_load_schema(), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "document"
        raise RegistrySchemaError(f"registry schema error at {location}: {first.message}")

    metadata = document["registry"]
    generated_at = _parse_datetime(metadata["generated_at"])
    expires_at = _parse_datetime(metadata["expires_at"])
    if expires_at <= generated_at:
        raise RegistrySchemaError("registry expires_at must be later than generated_at")

    ids = [item["id"] for item in document["models"]]
    duplicates = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
    if duplicates:
        raise RegistrySchemaError(f"duplicate model ids: {', '.join(duplicates)}")


def _http_fetch(url: str, etag: str | None, timeout: float) -> HttpResult:
    headers = {
        "Accept": "application/json",
        "User-Agent": "autonomyfit-registry-client/0.2",
    }
    if etag:
        headers["If-None-Match"] = etag
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResult(
                status=response.status,
                body=response.read(),
                etag=response.headers.get("ETag"),
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return HttpResult(status=304, etag=exc.headers.get("ETag") or etag)
        raise RegistryUnavailableError(f"registry request failed with HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RegistryUnavailableError(f"registry request failed: {exc}") from exc


def verify_registry_signature(registry_bytes: bytes, bundle_bytes: bytes) -> None:
    try:
        from sigstore.errors import Error as SigstoreError
        from sigstore.models import Bundle
        from sigstore.verify import Verifier as SigstoreVerifier
        from sigstore.verify.policy import Identity

        bundle = Bundle.from_json(bundle_bytes)
        verifier = SigstoreVerifier.production(offline=False)
        policy = Identity(identity=EXPECTED_CERT_IDENTITY, issuer=EXPECTED_OIDC_ISSUER)
        verifier.verify_artifact(registry_bytes, bundle, policy)
    except (ImportError, ValueError) as exc:
        raise RegistryTrustError(f"Sigstore verification could not run: {exc}") from exc
    except SigstoreError as exc:
        raise RegistryTrustError(f"Sigstore verification failed: {exc}") from exc


def _model_from_v2(item: dict[str, Any]) -> ModelProfile:
    accuracy_records = item["evidence"].get("accuracy", [])
    accuracy = None
    if accuracy_records:
        primary = accuracy_records[0]
        accuracy = AccuracyMetric(
            name=primary["metric"],
            value=float(primary["value"]),
            dataset=primary.get("dataset"),
        )

    memory_records = item["evidence"].get("memory", [])
    memory_gb = None
    memory_scope = None
    if memory_records:
        primary_memory = memory_records[0]
        memory_gb = float(primary_memory["gb"])
        memory_scope = primary_memory.get("scope")

    input_info = item.get("input", {})
    input_size = input_info.get("size")
    if input_size is None and input_info.get("width") == input_info.get("height"):
        input_size = input_info.get("width")

    upstream = item["upstream"]
    compatibility = item["compatibility"]
    license_info = item["license"]
    verification = item["verification"]
    family = item["family"]
    modalities = item["modalities"]
    parameters = item["parameters"]

    return ModelProfile(
        id=item["id"],
        display_name=item["display_name"],
        family=family["name"],
        variant=family.get("variant"),
        task=item["task"],
        params_m=float(parameters["millions"]),
        flops_b=float(parameters["flops_b"]) if parameters.get("flops_b") is not None else None,
        input_size=int(input_size) if input_size is not None else None,
        source_id=upstream["source_id"],
        source_url=upstream["source_url"],
        source_revision=upstream.get("revision"),
        release_date=upstream.get("release_date"),
        last_checked=upstream.get("last_checked"),
        runtimes=tuple(compatibility["runtimes"]),
        supported_precisions=tuple(compatibility.get("precisions", [])),
        quantizations=tuple(compatibility.get("quantizations", [])),
        input_modalities=tuple(modalities["input"]),
        output_modalities=tuple(modalities["output"]),
        accuracy=accuracy,
        published_memory_gb=memory_gb,
        memory_scope=memory_scope,
        license_spdx=license_info.get("spdx"),
        license_status=license_info.get("status", "unknown"),
        license_source_url=license_info.get("source_url"),
        verification_status=verification["status"],
        last_verified=verification.get("last_verified"),
        benchmark_refs=tuple(item["evidence"].get("benchmark_refs", [])),
        notes=item.get("notes"),
    )


def models_from_registry(document: dict[str, Any]) -> list[ModelProfile]:
    validate_registry_document(document)
    return [_model_from_v2(item) for item in document["models"]]


class RegistryClient:
    def __init__(
        self,
        *,
        registry_url: str = DEFAULT_REGISTRY_URL,
        bundle_url: str = DEFAULT_BUNDLE_URL,
        cache_dir: Path | None = None,
        refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
        timeout: float = 5.0,
        fetcher: Fetcher = _http_fetch,
        verifier: Verifier = verify_registry_signature,
        now_fn: NowFn = _utc_now,
    ) -> None:
        self.registry_url = registry_url
        self.bundle_url = bundle_url
        env_cache = os.environ.get("AUTONOMYFIT_CACHE_DIR")
        self.cache_dir = (
            Path(env_cache).expanduser()
            if cache_dir is None and env_cache
            else cache_dir or user_cache_path("autonomyfit") / "registry"
        )
        self.refresh_seconds = refresh_seconds
        self.timeout = timeout
        self.fetcher = fetcher
        self.verifier = verifier
        self.now_fn = now_fn

    @property
    def registry_path(self) -> Path:
        return self.cache_dir / "registry-v2.json"

    @property
    def bundle_path(self) -> Path:
        return self.cache_dir / "registry-v2.json.sigstore.json"

    @property
    def cache_state_path(self) -> Path:
        return self.cache_dir / "cache-state.json"

    @property
    def security_state_path(self) -> Path:
        return self.cache_dir / "security-state.json"

    def _read_state(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_state(self, path: Path, value: dict[str, Any]) -> None:
        _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())

    def _read_cached(self) -> tuple[dict[str, Any], bytes, bytes] | None:
        try:
            registry_bytes = self.registry_path.read_bytes()
            bundle_bytes = self.bundle_path.read_bytes()
        except OSError:
            return None

        cache_state = self._read_state(self.cache_state_path)
        expected_digest = cache_state.get("sha256")
        if not expected_digest or expected_digest != _sha256(registry_bytes):
            return None
        try:
            document = _read_json_bytes(registry_bytes, "cached registry")
            validate_registry_document(document)
        except RegistryError:
            return None
        return document, registry_bytes, bundle_bytes

    def _fallback(self, warning: str | None = None) -> RegistrySnapshot:
        resource = files("autonomyfit.data").joinpath("fallback_registry.json")
        document = json.loads(resource.read_text(encoding="utf-8"))
        models = tuple(models_from_registry(document))
        metadata = document["registry"]
        now = self.now_fn()
        stale = _parse_datetime(metadata["expires_at"]) <= now
        fallback_warning = warning
        if stale:
            extra = "bundled fallback registry is past its freshness window"
            fallback_warning = f"{warning}; {extra}" if warning else extra
        return RegistrySnapshot(
            document=document,
            models=models,
            provenance=RegistryProvenance(
                source="bundled-fallback",
                registry_version=int(metadata["registry_version"]),
                generated_at=metadata["generated_at"],
                expires_at=metadata["expires_at"],
                loaded_at=_iso(now),
                signature_verified=False,
                stale=stale,
                warning=fallback_warning,
                registry_url=self.registry_url,
            ),
        )

    def _cache_snapshot(
        self,
        document: dict[str, Any],
        *,
        stale: bool,
        warning: str | None = None,
    ) -> RegistrySnapshot:
        metadata = document["registry"]
        cache_state = self._read_state(self.cache_state_path)
        return RegistrySnapshot(
            document=document,
            models=tuple(models_from_registry(document)),
            provenance=RegistryProvenance(
                source="cache",
                registry_version=int(metadata["registry_version"]),
                generated_at=metadata["generated_at"],
                expires_at=metadata["expires_at"],
                loaded_at=_iso(self.now_fn()),
                signature_verified=True,
                stale=stale,
                etag=cache_state.get("etag"),
                warning=warning,
                registry_url=self.registry_url,
            ),
        )

    def _check_monotonic(self, version: int, digest: str) -> None:
        security = self._read_state(self.security_state_path)
        highest = security.get("highest_seen_version")
        prior_digest = security.get("sha256")
        if isinstance(highest, int):
            if version < highest:
                raise RegistryRollbackError(
                    f"registry version {version} is older than trusted version {highest}"
                )
            if version == highest and prior_digest and prior_digest != digest:
                raise RegistryRollbackError(
                    "registry content changed without increasing registry_version"
                )

    def _accept_remote(
        self,
        registry_bytes: bytes,
        bundle_bytes: bytes,
        *,
        etag: str | None,
    ) -> RegistrySnapshot:
        self.verifier(registry_bytes, bundle_bytes)
        document = _read_json_bytes(registry_bytes, "remote registry")
        validate_registry_document(document)
        metadata = document["registry"]
        now = self.now_fn()
        generated_at = _parse_datetime(metadata["generated_at"])
        expires_at = _parse_datetime(metadata["expires_at"])
        if generated_at > now + MAX_CLOCK_SKEW:
            raise RegistryTrustError("registry generated_at is implausibly far in the future")
        if expires_at <= now:
            raise RegistryTrustError("remote registry is expired")

        version = int(metadata["registry_version"])
        digest = _sha256(registry_bytes)
        self._check_monotonic(version, digest)

        _atomic_write(self.registry_path, registry_bytes)
        _atomic_write(self.bundle_path, bundle_bytes)
        self._write_state(
            self.cache_state_path,
            {
                "etag": etag,
                "cached_at": _iso(now),
                "registry_version": version,
                "sha256": digest,
            },
        )
        self._write_state(
            self.security_state_path,
            {
                "highest_seen_version": version,
                "sha256": digest,
                "accepted_at": _iso(now),
            },
        )
        return RegistrySnapshot(
            document=document,
            models=tuple(models_from_registry(document)),
            provenance=RegistryProvenance(
                source="remote",
                registry_version=version,
                generated_at=metadata["generated_at"],
                expires_at=metadata["expires_at"],
                loaded_at=_iso(now),
                signature_verified=True,
                stale=False,
                etag=etag,
                registry_url=self.registry_url,
            ),
        )

    def _cache_age_seconds(self) -> float | None:
        state = self._read_state(self.cache_state_path)
        cached_at = state.get("cached_at")
        if not isinstance(cached_at, str):
            return None
        try:
            return max(0.0, (self.now_fn() - _parse_datetime(cached_at)).total_seconds())
        except RegistryError:
            return None

    def load(self, *, offline: bool = False, force: bool = False) -> RegistrySnapshot:
        cached = self._read_cached()
        if offline:
            if cached:
                document, _, _ = cached
                stale = _parse_datetime(document["registry"]["expires_at"]) <= self.now_fn()
                warning = "offline mode is using an expired cached registry" if stale else None
                return self._cache_snapshot(document, stale=stale, warning=warning)
            return self._fallback("offline mode has no verified registry cache")

        if cached and not force:
            document, _, _ = cached
            age = self._cache_age_seconds()
            expired = _parse_datetime(document["registry"]["expires_at"]) <= self.now_fn()
            if age is not None and age < self.refresh_seconds and not expired:
                return self._cache_snapshot(document, stale=False)

        cache_state = self._read_state(self.cache_state_path)
        etag = cache_state.get("etag") if cached else None
        try:
            remote = self.fetcher(self.registry_url, etag, self.timeout)
            if remote.status == 304:
                if not cached:
                    raise RegistryUnavailableError("registry returned 304 without a local cache")
                document, _, _ = cached
                expired = _parse_datetime(document["registry"]["expires_at"]) <= self.now_fn()
                if expired:
                    raise RegistryTrustError("cached registry expired while remote returned 304")
                cache_state["cached_at"] = _iso(self.now_fn())
                cache_state["etag"] = remote.etag or etag
                self._write_state(self.cache_state_path, cache_state)
                return self._cache_snapshot(document, stale=False)
            if remote.status != 200:
                raise RegistryUnavailableError(f"unexpected registry HTTP status {remote.status}")
            bundle = self.fetcher(self.bundle_url, None, self.timeout)
            if bundle.status != 200:
                raise RegistryUnavailableError(f"unexpected bundle HTTP status {bundle.status}")
            return self._accept_remote(remote.body, bundle.body, etag=remote.etag)
        except RegistryError as exc:
            if force:
                raise
            if cached:
                document, _, _ = cached
                stale = _parse_datetime(document["registry"]["expires_at"]) <= self.now_fn()
                warning = f"registry refresh failed: {exc}"
                if stale:
                    warning += "; using expired verified cache"
                return self._cache_snapshot(document, stale=stale, warning=warning)
            return self._fallback(f"registry refresh failed: {exc}")

    def update(self) -> RegistrySnapshot:
        return self.load(offline=False, force=True)

    def clear_cache(self) -> list[str]:
        removed: list[str] = []
        for path in (self.registry_path, self.bundle_path, self.cache_state_path):
            try:
                path.unlink()
                removed.append(str(path))
            except FileNotFoundError:
                pass
        return removed

    def status(self) -> dict[str, object]:
        cached = self._read_cached()
        cache_info: dict[str, object] | None = None
        if cached:
            document, _, _ = cached
            metadata = document["registry"]
            expires = _parse_datetime(metadata["expires_at"])
            cache_state = self._read_state(self.cache_state_path)
            cache_info = {
                "registry_version": metadata["registry_version"],
                "generated_at": metadata["generated_at"],
                "expires_at": metadata["expires_at"],
                "stale": expires <= self.now_fn(),
                "cached_at": cache_state.get("cached_at"),
                "etag": cache_state.get("etag"),
                "sha256": cache_state.get("sha256"),
            }
        fallback = self._fallback().document["registry"]
        security = self._read_state(self.security_state_path)
        return {
            "registry_url": self.registry_url,
            "cache_dir": str(self.cache_dir),
            "cache": cache_info,
            "fallback": {
                "registry_version": fallback["registry_version"],
                "generated_at": fallback["generated_at"],
                "expires_at": fallback["expires_at"],
            },
            "highest_seen_version": security.get("highest_seen_version"),
            "security_state_preserved_on_clear": True,
        }
