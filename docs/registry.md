# Registry architecture

AutonomyFit deliberately separates engine releases from model-data releases.

## Files

- `registry/source/registry-v2.json` is the maintained source document.
- `registry/published/registry-v2.json` is the client-facing artifact.
- `registry/published/registry-v2.json.sigstore.json` is its Sigstore verification bundle.
- `registry/discovery/config.json` defines discovery/source-quality policy.
- `registry/discovery/latest.json` is the latest meaningful discovery audit snapshot.
- `src/autonomyfit/data/fallback_registry.json` is the package bootstrap/offline snapshot.
- `src/autonomyfit/data/registry-v2.schema.json` defines Registry Schema v2.

A model-data change increments `registry.registry_version`. It does not change the Python
package version.

## Client selection

The client uses this order:

1. a fresh verified cache, without network access
2. a conditional remote request when the cache reaches its refresh interval
3. a newly downloaded registry only after signature, schema, freshness and rollback checks
4. the existing verified cache if refresh fails, with an explicit warning
5. the bundled fallback when no verified cache exists

`--offline` skips all network access and uses the cache or fallback.

## Trust

The publishing workflow signs the exact registry bytes with Sigstore keyless signing using
GitHub Actions OIDC. The client requires the certificate identity:

`https://github.com/sylvesterkaczmarek/autonomyfit/.github/workflows/registry-publish.yml@refs/heads/main`

and issuer:

`https://token.actions.githubusercontent.com`

A successful signature is necessary but not sufficient. The client also stores the highest
accepted registry version and digest. It rejects a lower version and rejects different content
that reuses the same version. The signed document contains generation and expiry times so a
stale/frozen registry is detectable.

Cache clearing does not erase this rollback-protection state.

## Discovery and approval

Stage 2 adds a separate discovery pipeline. Provider records do not directly become
recommendation records.

The discovery lifecycle is:

`DISCOVERED -> NORMALIZED -> SOURCE_VERIFIED -> COMPATIBILITY_VERIFIED -> BENCHMARKED`

`DEPRECATED` is an audit tombstone for a discovery identity that disappears upstream.

Automatic registry promotion requires at least `SOURCE_VERIFIED`. Missing licence, missing
revision, unsupported task, missing parameter count, incomplete runtime metadata or an
unapproved publisher prevents automatic promotion.

The signed Registry v2 remains the only input to normal model selection. Discovery audit data
does not bypass the registry trust boundary.

See [discovery.md](discovery.md).

## Failure semantics

Remote JSON, schema or signature failures never overwrite the last accepted cache. A stale
verified cache can still be used for availability, but it is marked stale and carries a warning.
The bundled fallback is trusted as part of the installed package, not as a live signed registry.

Discovery failures are also fail-closed: malformed/incomplete upstream records do not replace
trusted registry entries.

## Why Sigstore

A full TUF repository provides strong multi-role update security, particularly when offline
root/targets keys and a frequently rotated online timestamp role can be operated correctly.
AutonomyFit does not currently have that key-management infrastructure.

Sigstore lets the registry publisher use short-lived GitHub OIDC identity instead of a
repository secret or long-lived private signing key, while recording signing events in the
transparency log.

If the registry later uses multiple independent distribution mirrors or managed signing
infrastructure, a TUF role hierarchy can be layered on without changing the Registry v2 data
model.

## Updating registry data manually

For curated corrections, change `registry/source/registry-v2.json`, increment
`registry_version`, refresh signed freshness timestamps, validate it and push the data change.
The registry publisher signs and updates `registry/published/` independently of PyPI.

For normal model discovery, the scheduled refresh workflow performs this process
automatically.

The Python package only needs a release when code or supported schema behavior changes.
