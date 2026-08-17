# Continuous model discovery

AutonomyFit Stage 2 turns registry maintenance into a provider-driven metadata pipeline.

## Provider architecture

Discovery is isolated from the scoring engine behind `ModelSourceAdapter`.

Implemented adapters:

| Adapter | Machine-readable source | Role |
|---|---|---|
| `HuggingFaceAdapter` | Hugging Face Hub model APIs | Broad discovery for object detection and compact VLMs. |
| `NvidiaAdapter` | NVIDIA's official Hugging Face publisher feed | Higher-priority NVIDIA discovery. |
| `UltralyticsAdapter` | Official GitHub release and repository-content APIs | Detects official YOLO family/version signals. |
| `VendorGitHubAdapter` | GitHub Releases REST API | Generic vendor release/version signals. |

The pipeline uses provider metadata only. It does not import model repositories, execute
remote code or download model weights.

## Hugging Face metadata

The Hub adapter uses model-list and model-detail API data for:

- pipeline task
- repository identity
- exact repository SHA
- creation/update timestamps
- downloads and likes
- library name and tags
- model-card licence
- base-model relationship
- `new_version` relationship
- safetensors parameter metadata when exposed by the Hub

The selection engine supports ten task categories, but scheduled Hub promotion currently covers object detection and compact VLMs only. The other task categories use curated registry entries until equally conservative provider-specific discovery and normalization rules exist.

Current automatic promotion caps are intentionally edge-oriented:

- VLM: at most 4 billion parameters
- detection: at most 2 billion parameters

Larger models can still be observed in discovery data, but are not promoted into the
edge-selection registry.

## NVIDIA

NVIDIA NGC was evaluated as an authoritative vendor source. Current NGC documentation
describes catalog models, release metadata, signed-model verification and model APIs. It also
documents NVIDIA model signing and the NVIDIA model-signing root certificate.

Stage 2 does not depend on an undocumented broad NGC search endpoint. The `NvidiaAdapter`
therefore uses the official `nvidia` publisher on the Hugging Face Hub for broad automated
discovery. NGC remains a source for future model-specific enrichment and signature
corroboration when a stable catalog-list interface is appropriate.

## Ultralytics

Ultralytics discovery uses GitHub's machine-readable REST endpoints against the official
`ultralytics/ultralytics` repository:

1. latest published release
2. model configuration directory listing at that release

This detects new official YOLO family signals without scraping documentation HTML. Family
signals remain discovery-only until parameter/evidence metadata is sufficient for promotion.

## Lifecycle

Discovery and approval are separate.

### `DISCOVERED`

The upstream record exists but required metadata is incomplete. Common reasons include a
missing parameter count or missing runtime metadata.

### `NORMALIZED`

AutonomyFit has a stable identity and revision mapping, but source-quality requirements for
automatic promotion are not met.

### `SOURCE_VERIFIED`

The candidate has:

- an approved official publisher
- an exact upstream revision
- a published licence identifier
- supported task
- parameter count
- at least one usable runtime

Only this state can be automatically promoted into Registry v2.

### `COMPATIBILITY_VERIFIED`

Compatibility has additional direct evidence. This remains a stronger state than source
verification and is not granted merely because a model was discovered.

### `BENCHMARKED`

Exact hardware/runtime/precision benchmark evidence exists.

### `DEPRECATED`

A previously observed discovery identity is no longer returned by the current provider set.
The record becomes a discovery tombstone. It is not silently deleted from the signed registry.

## Canonical identity and revisions

For Hub-backed records the canonical discovery identity is the case-normalized Hub repository
ID. NVIDIA records discovered through both the general Hub feed and the NVIDIA feed therefore
collapse to the same identity.

Each upstream revision gets a stable `revision_identity` derived from:

```text
canonical identity + exact upstream revision
```

Deduplication then chooses the strongest source according to policy:

1. Ultralytics official source
2. NVIDIA official source
3. configured vendor GitHub source
4. general Hugging Face source

Aliases from lower-priority duplicate records are retained in the discovery manifest.

Existing Registry v2 IDs are preserved when a candidate source URL already exists. This
prevents model IDs from changing because another provider also discovered the model.

## Licence handling

Missing licence metadata never defaults to MIT.

Discovery classifies licence metadata as:

- `published`
- `restricted`
- `unknown`

Non-commercial/research-only/custom licence signals are flagged. A missing licence prevents
automatic source verification and registry promotion.

Licence status is metadata, not legal advice. Users must still review the upstream terms for
their intended use.

## Scheduled refresh

`.github/workflows/registry-refresh.yml` runs every day at 05:17 UTC and can also be
triggered manually.

A refresh:

1. discovers provider records
2. normalizes and deduplicates them
3. preserves source/revision/provenance
4. creates deprecation tombstones for disappeared discovery records
5. applies approval gates
6. deterministically merges approved records into Registry v2
7. validates the generated registry
8. runs Ruff and the discovery test suite
9. commits only meaningful changes

Download/like counters are retained as provenance but ignored when deciding whether a commit
is meaningful. This avoids daily commits caused only by popularity counters.

If no model/discovery semantics change, the workflow makes no commit. Registry freshness is
renewed on a longer interval so freeze detection remains meaningful without daily churn.

When Registry v2 itself changes, the workflow calls the existing reusable
`registry-publish.yml`. That workflow signs the exact registry bytes with Sigstore using the
same GitHub Actions OIDC identity accepted by the client.

The scheduled job never publishes a Python package.

## Discovery audit data

`registry/discovery/latest.json` records the normalized discovery view when meaningful state
changes. It includes:

- provider
- upstream ID
- publisher
- source/evidence URL
- exact revision
- revision identity
- release/update timestamps
- parameter count
- library/runtime metadata
- licence metadata
- aliases
- base-model/new-version relationships
- lifecycle
- source-staleness flag
- warnings

This file is audit/debug data. The client-facing trust boundary remains the signed Registry v2
artifact.

## Failure behavior

Discovery fails closed when every provider fails.

A single provider failure does not allow that provider to overwrite trusted data. Other
providers can continue, and provider-health warnings are surfaced in discovery audit data.

Malformed/incomplete records remain unpromoted.

The signed registry is only updated after schema validation and the existing Stage 1 trust
checks.
