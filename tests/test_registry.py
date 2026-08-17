from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from autonomyfit.registry import (
    HttpResult,
    RegistryClient,
    RegistryRollbackError,
    RegistrySchemaError,
    RegistryTrustError,
    models_from_registry,
    validate_registry_document,
)

NOW = datetime(2026, 8, 16, 9, 35, tzinfo=timezone.utc)


def _fallback_document() -> dict:
    path = Path(__file__).parents[1] / "src/autonomyfit/data/fallback_registry.json"
    return json.loads(path.read_text())


def _document(version: int = 2, *, expires_delta: timedelta = timedelta(days=30)) -> dict:
    value = _fallback_document()
    value["registry"]["registry_version"] = version
    value["registry"]["generated_at"] = "2026-08-16T09:35:00Z"
    expiry = NOW + expires_delta
    value["registry"]["expires_at"] = expiry.isoformat().replace("+00:00", "Z")
    return value


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _client(tmp_path, responses, *, verifier=None, refresh_seconds=0):
    calls = []

    def fetcher(url, etag, timeout):
        calls.append((url, etag))
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def good_verifier(data, bundle):
        assert data
        assert bundle == b"bundle"

    client = RegistryClient(
        cache_dir=tmp_path,
        fetcher=fetcher,
        verifier=verifier or good_verifier,
        now_fn=lambda: NOW,
        refresh_seconds=refresh_seconds,
    )
    return client, calls


def _remote_pair(document, etag='"v2"'):
    return [HttpResult(200, _bytes(document), etag), HttpResult(200, b"bundle")]


def test_registry_schema_accepts_fallback():
    document = _fallback_document()
    validate_registry_document(document)
    assert len(models_from_registry(document)) == 45


def test_valid_remote_registry_is_cached_and_verified(tmp_path):
    client, calls = _client(tmp_path, _remote_pair(_document()))
    snapshot = client.update()
    assert snapshot.provenance.source == "remote"
    assert snapshot.provenance.signature_verified is True
    assert snapshot.provenance.registry_version == 2
    assert client.registry_path.exists()
    assert len(calls) == 2


def test_fresh_cache_avoids_network(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document()), refresh_seconds=3600)
    client.update()
    client.fetcher = lambda *_: (_ for _ in ()).throw(AssertionError("network used"))
    snapshot = client.load()
    assert snapshot.provenance.source == "cache"


def test_conditional_304_uses_verified_cache(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document()), refresh_seconds=0)
    client.update()
    client.fetcher = lambda url, etag, timeout: HttpResult(304, etag=etag)
    snapshot = client.load()
    assert snapshot.provenance.source == "cache"
    assert snapshot.provenance.stale is False


def test_offline_uses_cache_without_network(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document()))
    client.update()
    client.fetcher = lambda *_: (_ for _ in ()).throw(AssertionError("network used"))
    assert client.load(offline=True).provenance.source == "cache"


def test_offline_without_cache_uses_bundled_fallback(tmp_path):
    client, _ = _client(tmp_path, [])
    snapshot = client.load(offline=True)
    assert snapshot.provenance.source == "bundled-fallback"
    assert "no verified registry cache" in (snapshot.provenance.warning or "")


def test_invalid_json_never_replaces_cache(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document()))
    accepted = client.update()
    original = client.registry_path.read_bytes()
    responses = [HttpResult(200, b"not-json", '"bad"'), HttpResult(200, b"bundle")]
    client.fetcher = lambda url, etag, timeout: responses.pop(0)
    snapshot = client.load(force=False)
    assert snapshot.provenance.source == "cache"
    assert client.registry_path.read_bytes() == original
    assert snapshot.provenance.registry_version == accepted.provenance.registry_version


def test_invalid_schema_falls_back_without_cache(tmp_path):
    bad = _document()
    bad["models"][0].pop("family")
    client, _ = _client(tmp_path, _remote_pair(bad))
    snapshot = client.load()
    assert snapshot.provenance.source == "bundled-fallback"
    assert "schema" in (snapshot.provenance.warning or "")


def test_incompatible_schema_is_rejected():
    bad = _fallback_document()
    bad["schema_version"] = 99
    with pytest.raises(RegistrySchemaError, match="unsupported registry schema"):
        validate_registry_document(bad)


def test_signature_failure_does_not_replace_cache(tmp_path):
    good_client, _ = _client(tmp_path, _remote_pair(_document()))
    good_client.update()
    original = good_client.registry_path.read_bytes()

    def reject(data, bundle):
        raise RegistryTrustError("bad signature")

    responses = _remote_pair(_document(version=3))
    good_client.fetcher = lambda url, etag, timeout: responses.pop(0)
    good_client.verifier = reject
    snapshot = good_client.load()
    assert snapshot.provenance.source == "cache"
    assert "bad signature" in (snapshot.provenance.warning or "")
    assert good_client.registry_path.read_bytes() == original


def test_force_update_surfaces_signature_failure(tmp_path):
    def reject(data, bundle):
        raise RegistryTrustError("bad signature")

    client, _ = _client(tmp_path, _remote_pair(_document()), verifier=reject)
    with pytest.raises(RegistryTrustError, match="bad signature"):
        client.update()


def test_rollback_is_rejected(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document(version=5)))
    client.update()
    responses = _remote_pair(_document(version=4))
    client.fetcher = lambda url, etag, timeout: responses.pop(0)
    with pytest.raises(RegistryRollbackError, match="older than trusted"):
        client.update()


def test_same_version_different_content_is_rejected(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document(version=5)))
    client.update()
    changed = _document(version=5)
    changed["models"][0]["display_name"] = "Changed without version"
    responses = _remote_pair(changed)
    client.fetcher = lambda url, etag, timeout: responses.pop(0)
    with pytest.raises(RegistryRollbackError, match="without increasing"):
        client.update()


def test_expired_remote_is_rejected(tmp_path):
    expired = _document()
    expired["registry"]["generated_at"] = "2026-08-15T09:35:00Z"
    expired["registry"]["expires_at"] = "2026-08-16T09:34:59Z"
    client, _ = _client(tmp_path, _remote_pair(expired))
    with pytest.raises(RegistryTrustError, match="expired"):
        client.update()


def test_stale_cached_registry_is_visible_offline(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document()))
    client.update()
    client.now_fn = lambda: NOW + timedelta(days=31)
    snapshot = client.load(offline=True)
    assert snapshot.provenance.stale is True
    assert "expired cached registry" in (snapshot.provenance.warning or "")


def test_network_failure_uses_verified_cache_with_warning(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document()), refresh_seconds=0)
    client.update()
    from autonomyfit.registry import RegistryUnavailableError
    client.fetcher = lambda *_: (_ for _ in ()).throw(RegistryUnavailableError("offline"))
    snapshot = client.load()
    assert snapshot.provenance.source == "cache"
    assert "refresh failed" in (snapshot.provenance.warning or "")


def test_corrupt_cache_digest_is_not_used(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document()))
    client.update()
    client.registry_path.write_text("{}")
    snapshot = client.load(offline=True)
    assert snapshot.provenance.source == "bundled-fallback"


def test_clear_cache_preserves_security_state(tmp_path):
    client, _ = _client(tmp_path, _remote_pair(_document(version=7)))
    client.update()
    assert client.security_state_path.exists()
    client.clear_cache()
    assert not client.registry_path.exists()
    assert not client.cache_state_path.exists()
    assert client.security_state_path.exists()
    assert client.status()["highest_seen_version"] == 7


def test_future_generated_time_is_rejected(tmp_path):
    future = _document()
    future["registry"]["generated_at"] = "2026-08-16T10:00:00Z"
    client, _ = _client(tmp_path, _remote_pair(future))
    with pytest.raises(RegistryTrustError, match="future"):
        client.update()


def test_duplicate_ids_are_rejected():
    document = _document()
    document["models"].append(document["models"][0])
    with pytest.raises(RegistrySchemaError, match="duplicate model ids"):
        validate_registry_document(document)

def test_cache_cannot_override_preserved_security_state(tmp_path):
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
    assert snapshot.provenance.source == "bundled-fallback"
