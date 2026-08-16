from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

SUPPORTED_TASKS = {"detection", "vlm"}
HF_TASKS = {
    "object-detection": "detection",
    "image-text-to-text": "vlm",
    "visual-question-answering": "vlm",
}
OPEN_LICENSE_PREFIXES = (
    "apache-",
    "mit",
    "bsd-",
    "cc-by-",
    "mpl-",
)
RESTRICTED_LICENSE_MARKERS = (
    "-nc",
    "non-commercial",
    "noncommercial",
    "research-only",
    "research only",
)
SOURCE_PRIORITY = {
    "ultralytics": 100,
    "nvidia": 95,
    "vendor-github": 90,
    "huggingface": 70,
}
MAX_METADATA_BYTES = 5 * 1024 * 1024


class DiscoveryError(RuntimeError):
    """A provider or normalization error."""


@dataclass(frozen=True)
class DiscoveryCandidate:
    provider: str
    upstream_id: str
    source_url: str
    publisher: str
    display_name: str
    family: str
    variant: str | None
    task: str | None
    revision: str | None
    release_date: str | None
    last_modified: str | None
    params_m: float | None
    license_id: str | None
    library: str | None
    runtimes: tuple[str, ...]
    precisions: tuple[str, ...] = ()
    quantizations: tuple[str, ...] = ()
    downloads: int | None = None
    likes: int | None = None
    aliases: tuple[str, ...] = ()
    base_model: str | None = None
    base_model_relation: str | None = None
    new_version: str | None = None
    evidence_url: str | None = None
    ingestion_method: str = "api"
    official_publisher: bool = False
    deprecated: bool = False
    metadata_complete: bool = True
    warnings: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def identity_key(self) -> str:
        if self.provider in {"huggingface", "nvidia"}:
            return f"hf:{self.upstream_id.casefold()}"
        return f"{self.provider}:{self.upstream_id.casefold()}"

    @property
    def revision_identity(self) -> str:
        value = f"{self.identity_key}@{self.revision or 'unknown'}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    def lifecycle(self) -> str:
        if self.deprecated:
            return "DEPRECATED"
        if not self.metadata_complete:
            return "DISCOVERED"
        if self.official_publisher and self.revision and self.license_id:
            return "SOURCE_VERIFIED"
        return "NORMALIZED"

    def to_manifest_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("raw", None)
        value["identity_key"] = self.identity_key
        value["revision_identity"] = self.revision_identity
        value["lifecycle"] = self.lifecycle()
        value["source_priority"] = SOURCE_PRIORITY.get(self.provider, 0)
        value["aliases"] = sorted(set(value["aliases"]))
        value["runtimes"] = sorted(set(value["runtimes"]))
        value["precisions"] = sorted(set(value["precisions"]))
        value["quantizations"] = sorted(set(value["quantizations"]))
        value["warnings"] = list(value["warnings"])
        return value


class ModelSourceAdapter(Protocol):
    name: str

    def discover(self, now: datetime) -> list[DiscoveryCandidate]:
        """Return source records without executing model repository code."""


class JsonTransport:
    def get_json(self, url: str, *, timeout: float = 15.0) -> Any:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "autonomyfit-discovery/0.3",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(MAX_METADATA_BYTES + 1)
                if len(body) > MAX_METADATA_BYTES:
                    raise DiscoveryError(
                        f"metadata response exceeded {MAX_METADATA_BYTES} bytes for {url}"
                    )
                return json.loads(body.decode("utf-8"))
        except DiscoveryError:
            raise
        except (
            urllib.error.URLError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise DiscoveryError(f"request failed for {url}: {exc}") from exc


def _parse_dt(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _date_only(value: str | None) -> str | None:
    return value[:10] if value and len(value) >= 10 else None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return re.sub(r"-+", "-", slug)


def _family_variant(name: str) -> tuple[str, str | None]:
    cleaned = name.rsplit("/", 1)[-1].replace("_", "-")
    tokens = [token for token in cleaned.split("-") if token]
    variant_start = None
    for index, token in enumerate(tokens):
        if re.fullmatch(r"\d+(?:\.\d+)?[mb]", token, re.IGNORECASE):
            variant_start = index
            break
        if token.casefold() in {
            "nano",
            "small",
            "medium",
            "large",
            "base",
            "tiny",
            "mini",
        }:
            variant_start = index
            break
    if variant_start is None or variant_start == 0:
        return cleaned, None
    return "-".join(tokens[:variant_start]), "-".join(tokens[variant_start:])


def _extract_params_m(payload: dict[str, Any]) -> float | None:
    safetensors = payload.get("safetensors")
    if isinstance(safetensors, dict):
        direct = safetensors.get("total")
        if isinstance(direct, (int, float)) and direct > 0:
            return float(direct) / 1_000_000
        counts = safetensors.get("parameters") or safetensors.get("parameter_count")
        if isinstance(counts, dict):
            values = [value for value in counts.values() if isinstance(value, (int, float))]
            if values:
                total = sum(values)
                if total > 0:
                    return float(total) / 1_000_000

    config = payload.get("config")
    if isinstance(config, dict):
        for key in ("num_parameters", "num_params", "parameter_count"):
            value = config.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value) / 1_000_000
    return None


def _license_status(license_id: str | None) -> tuple[str, str | None]:
    if not license_id:
        return "unknown", "licence metadata missing upstream"
    normalized = license_id.casefold()
    if any(marker in normalized for marker in RESTRICTED_LICENSE_MARKERS):
        return "restricted", f"upstream licence requires review: {license_id}"
    if normalized == "other":
        return "restricted", "upstream uses a custom licence"
    if normalized.startswith(OPEN_LICENSE_PREFIXES):
        return "published", None
    return "published", f"non-standard licence identifier: {license_id}"


def _runtime_tags(payload: dict[str, Any], task: str | None) -> tuple[str, ...]:
    tags = {str(tag).casefold() for tag in payload.get("tags") or []}
    library = str(payload.get("library_name") or "").casefold()
    runtimes: set[str] = set()
    if library in {"transformers", "timm", "pytorch", "ultralytics"}:
        runtimes.add("pytorch")
    if library == "transformers":
        runtimes.add("transformers")
    if "onnx" in tags or "onnxruntime" in tags:
        runtimes.add("onnx")
    if "tensorrt" in tags:
        runtimes.add("tensorrt")
    if "coreml" in tags:
        runtimes.add("coreml")
    return tuple(sorted(runtimes))


class HuggingFaceAdapter:
    name = "huggingface"

    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        trusted_publishers: set[str] | None = None,
        limit_per_task: int = 25,
        author: str | None = None,
        provider_name: str = "huggingface",
    ) -> None:
        self.transport = transport or JsonTransport()
        self.trusted_publishers = {
            item.casefold()
            for item in (
                trusted_publishers
                or {
                    "HuggingFaceTB",
                    "google",
                    "microsoft",
                    "Qwen",
                    "nvidia",
                    "facebook",
                    "IDEA-Research",
                    "PekingU",
                    "ultralytics",
                }
            )
        }
        self.limit_per_task = limit_per_task
        self.author = author
        self.name = provider_name

    def _list_url(self, pipeline_tag: str) -> str:
        params = {
            "filter": pipeline_tag,
            "sort": "trendingScore",
            "direction": "-1",
            "limit": str(self.limit_per_task),
        }
        if self.author:
            params["author"] = self.author
        return "https://huggingface.co/api/models?" + urllib.parse.urlencode(params)

    def _detail_url(self, model_id: str) -> str:
        return "https://huggingface.co/api/models/" + urllib.parse.quote(
            model_id,
            safe="/",
        )

    def discover(self, now: datetime) -> list[DiscoveryCandidate]:
        del now
        records: dict[str, DiscoveryCandidate] = {}
        for pipeline_tag, task in HF_TASKS.items():
            listing = self.transport.get_json(self._list_url(pipeline_tag))
            if not isinstance(listing, list):
                raise DiscoveryError("Hugging Face model list was not an array")
            for summary in listing:
                if not isinstance(summary, dict) or not summary.get("id"):
                    continue
                model_id = str(summary["id"])
                detail = self.transport.get_json(self._detail_url(model_id))
                if not isinstance(detail, dict):
                    continue
                candidate = self._candidate(detail, task)
                existing = records.get(candidate.identity_key)
                if existing is None or (candidate.downloads or 0) > (existing.downloads or 0):
                    records[candidate.identity_key] = candidate
        return sorted(records.values(), key=lambda item: item.identity_key)

    def _candidate(self, payload: dict[str, Any], task: str) -> DiscoveryCandidate:
        model_id = str(payload.get("id") or payload.get("modelId") or "")
        if not model_id:
            raise DiscoveryError("Hugging Face model record has no id")
        publisher = model_id.split("/", 1)[0]
        card = payload.get("cardData")
        if not isinstance(card, dict):
            card = {}
        pipeline_tag = payload.get("pipeline_tag") or card.get("pipeline_tag")
        mapped_task = HF_TASKS.get(str(pipeline_tag), task)
        created = _parse_dt(payload.get("createdAt"))
        modified = _parse_dt(payload.get("lastModified"))
        family, variant = _family_variant(model_id)
        license_id = card.get("license")
        if license_id is not None:
            license_id = str(license_id)
        warnings: list[str] = []
        _, license_warning = _license_status(license_id)
        if license_warning:
            warnings.append(license_warning)
        params_m = _extract_params_m(payload)
        if params_m is None:
            warnings.append("parameter count unavailable from Hub metadata")
        base_model = card.get("base_model")
        if isinstance(base_model, list):
            base_model = ",".join(str(item) for item in base_model)
        new_version = card.get("new_version")
        source_url = f"https://huggingface.co/{model_id}"
        runtimes = _runtime_tags(payload, mapped_task)
        return DiscoveryCandidate(
            provider=self.name,
            upstream_id=model_id,
            source_url=source_url,
            publisher=publisher,
            display_name=str(card.get("model_name") or model_id.rsplit("/", 1)[-1]),
            family=family,
            variant=variant,
            task=mapped_task,
            revision=str(payload.get("sha")) if payload.get("sha") else None,
            release_date=_date_only(created),
            last_modified=modified,
            params_m=params_m,
            license_id=license_id,
            library=str(payload.get("library_name")) if payload.get("library_name") else None,
            runtimes=runtimes,
            downloads=(
                int(payload["downloads"])
                if isinstance(payload.get("downloads"), int)
                else None
            ),
            likes=int(payload["likes"]) if isinstance(payload.get("likes"), int) else None,
            aliases=(model_id, model_id.rsplit("/", 1)[-1]),
            base_model=str(base_model) if base_model else None,
            base_model_relation=(
                str(card.get("base_model_relation"))
                if card.get("base_model_relation")
                else None
            ),
            new_version=str(new_version) if new_version else None,
            evidence_url=source_url,
            ingestion_method="huggingface-hub-api",
            official_publisher=publisher.casefold() in self.trusted_publishers,
            metadata_complete=params_m is not None and bool(runtimes),
            warnings=tuple(warnings),
            raw=payload,
        )


class NvidiaAdapter(HuggingFaceAdapter):
    name = "nvidia"

    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        limit_per_task: int = 20,
    ) -> None:
        super().__init__(
            transport=transport,
            trusted_publishers={"nvidia"},
            limit_per_task=limit_per_task,
            author="nvidia",
            provider_name="nvidia",
        )


class UltralyticsAdapter:
    name = "ultralytics"

    def __init__(self, *, transport: JsonTransport | None = None) -> None:
        self.transport = transport or JsonTransport()

    def discover(self, now: datetime) -> list[DiscoveryCandidate]:
        del now
        release = self.transport.get_json(
            "https://api.github.com/repos/ultralytics/ultralytics/releases/latest"
        )
        if not isinstance(release, dict):
            raise DiscoveryError("Ultralytics release response was not an object")
        tag = str(release.get("tag_name") or "")
        if not tag:
            raise DiscoveryError("Ultralytics latest release has no tag")
        params = urllib.parse.urlencode({"ref": tag})
        contents = self.transport.get_json(
            "https://api.github.com/repos/ultralytics/ultralytics/"
            f"contents/ultralytics/cfg/models?{params}"
        )
        if not isinstance(contents, list):
            raise DiscoveryError("Ultralytics model directory response was not an array")

        candidates: list[DiscoveryCandidate] = []
        for entry in contents:
            if not isinstance(entry, dict) or entry.get("type") != "dir":
                continue
            family = str(entry.get("name") or "")
            if not re.fullmatch(r"yolo\d+", family, flags=re.IGNORECASE):
                continue
            source_url = str(
                entry.get("html_url")
                or "https://github.com/ultralytics/ultralytics"
            )
            candidates.append(
                DiscoveryCandidate(
                    provider=self.name,
                    upstream_id=family,
                    source_url=source_url,
                    publisher="ultralytics",
                    display_name=family.upper(),
                    family=family.upper(),
                    variant=None,
                    task="detection",
                    revision=tag,
                    release_date=_date_only(_parse_dt(release.get("published_at"))),
                    last_modified=_parse_dt(release.get("published_at")),
                    params_m=None,
                    license_id="AGPL-3.0-only",
                    library="ultralytics",
                    runtimes=("pytorch", "onnx", "tensorrt"),
                    aliases=(family,),
                    evidence_url=source_url,
                    ingestion_method="github-rest-release-and-contents",
                    official_publisher=True,
                    metadata_complete=False,
                    warnings=(
                        "family signal only; parameter evidence required before promotion",
                    ),
                )
            )
        return sorted(candidates, key=lambda item: item.identity_key)


class VendorGitHubAdapter:
    name = "vendor-github"

    def __init__(
        self,
        repositories: list[dict[str, str]],
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self.repositories = repositories
        self.transport = transport or JsonTransport()

    def discover(self, now: datetime) -> list[DiscoveryCandidate]:
        del now
        records: list[DiscoveryCandidate] = []
        for spec in self.repositories:
            repo = spec["repository"]
            release = self.transport.get_json(
                f"https://api.github.com/repos/{repo}/releases/latest"
            )
            if not isinstance(release, dict) or not release.get("tag_name"):
                continue
            family = spec["family"]
            task = spec.get("task")
            records.append(
                DiscoveryCandidate(
                    provider=self.name,
                    upstream_id=repo,
                    source_url=f"https://github.com/{repo}",
                    publisher=spec["publisher"],
                    display_name=family,
                    family=family,
                    variant=None,
                    task=task if task in SUPPORTED_TASKS else None,
                    revision=str(release["tag_name"]),
                    release_date=_date_only(_parse_dt(release.get("published_at"))),
                    last_modified=_parse_dt(release.get("published_at")),
                    params_m=None,
                    license_id=spec.get("license"),
                    library=None,
                    runtimes=(),
                    aliases=(repo, family),
                    new_version=str(release["tag_name"]),
                    evidence_url=str(
                        release.get("html_url") or f"https://github.com/{repo}/releases"
                    ),
                    ingestion_method="github-rest-releases",
                    official_publisher=True,
                    metadata_complete=False,
                    warnings=("release signal only; model metadata enrichment required",),
                )
            )
        return records


def deduplicate_candidates(
    candidates: list[DiscoveryCandidate],
) -> list[DiscoveryCandidate]:
    grouped: dict[str, list[DiscoveryCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.identity_key, []).append(candidate)

    output: list[DiscoveryCandidate] = []
    for identity, group in sorted(grouped.items()):
        del identity
        group.sort(
            key=lambda item: (
                item.official_publisher,
                SOURCE_PRIORITY.get(item.provider, 0),
                item.params_m is not None,
                item.revision is not None,
                item.downloads or 0,
            ),
            reverse=True,
        )
        primary = group[0]
        aliases = set(primary.aliases)
        warnings = list(primary.warnings)
        for duplicate in group[1:]:
            aliases.update(duplicate.aliases)
            aliases.add(duplicate.upstream_id)
            warnings.append(
                f"deduplicated corroborating source {duplicate.provider}:{duplicate.upstream_id}"
            )
        output.append(
            DiscoveryCandidate(
                **{
                    **asdict(primary),
                    "aliases": tuple(sorted(aliases)),
                    "warnings": tuple(dict.fromkeys(warnings)),
                    "raw": primary.raw,
                }
            )
        )
    return output


def _existing_index(document: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_id = {str(item["id"]): item for item in document.get("models", [])}
    by_url = {
        str(item["upstream"]["source_url"]).rstrip("/").casefold(): str(item["id"])
        for item in document.get("models", [])
        if isinstance(item.get("upstream"), dict) and item["upstream"].get("source_url")
    }
    return by_id, by_url


def _verification_rank(value: str) -> int:
    return {
        "discovered": 0,
        "source_verified": 1,
        "compatibility_verified": 2,
        "benchmarked": 3,
    }.get(value, -1)


def _candidate_model_id(
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
    return candidate_id


def candidate_to_registry_model(
    candidate: DiscoveryCandidate,
    *,
    model_id: str,
    checked_at: str,
) -> dict[str, Any] | None:
    if candidate.task not in SUPPORTED_TASKS:
        return None
    if candidate.params_m is None or candidate.params_m <= 0:
        return None
    if candidate.task == "vlm" and candidate.params_m > 4_000:
        return None
    if candidate.task == "detection" and candidate.params_m > 2_000:
        return None
    if not candidate.runtimes:
        return None
    if not (
        candidate.official_publisher
        and candidate.revision
        and candidate.license_id
    ):
        return None

    license_status, license_warning = _license_status(candidate.license_id)
    verification = "source_verified"
    notes = list(candidate.warnings)
    if license_warning:
        notes.append(license_warning)
    note = "; ".join(dict.fromkeys(notes)) if notes else None

    if candidate.task == "detection":
        modalities = {"input": ["image"], "output": ["object-detections"]}
        input_info = {
            "kind": "image",
            "size": None,
            "width": None,
            "height": None,
            "notes": "Input shape was not inferred during metadata-only discovery.",
        }
    else:
        modalities = {"input": ["image", "text"], "output": ["text"]}
        input_info = {
            "kind": "multimodal",
            "size": None,
            "width": None,
            "height": None,
            "notes": "Image/text model discovered from upstream metadata.",
        }

    source_id = f"{candidate.provider}:{candidate.upstream_id}"
    source_url = candidate.evidence_url or candidate.source_url
    return {
        "id": model_id,
        "display_name": candidate.display_name,
        "family": {"name": candidate.family, "variant": candidate.variant},
        "task": candidate.task,
        "modalities": modalities,
        "upstream": {
            "source_id": source_id,
            "source_url": candidate.source_url,
            "revision": candidate.revision,
            "release_date": candidate.release_date,
            "last_checked": checked_at,
        },
        "parameters": {"millions": round(candidate.params_m, 6), "flops_b": None},
        "input": input_info,
        "compatibility": {
            "runtimes": sorted(set(candidate.runtimes)),
            "precisions": sorted(set(candidate.precisions)),
            "quantizations": sorted(set(candidate.quantizations)),
        },
        "license": {
            "spdx": candidate.license_id,
            "status": license_status,
            "source_url": source_url if candidate.license_id else None,
            "note": license_warning,
        },
        "evidence": {
            "accuracy": [],
            "memory": [],
            "compatibility": [
                {"source_id": source_id, "source_url": source_url}
            ],
            "benchmark_refs": [],
        },
        "verification": {
            "status": verification,
            "last_verified": checked_at,
        },
        "notes": note,
    }


def _merge_existing(
    existing: dict[str, Any],
    discovered: dict[str, Any],
) -> dict[str, Any]:
    output = json.loads(json.dumps(existing))
    old_source = str(output["upstream"].get("source_url") or "")
    new_source = str(discovered["upstream"].get("source_url") or "")
    same_source = old_source.rstrip("/").casefold() == new_source.rstrip("/").casefold()

    if same_source:
        old_revision = output["upstream"].get("revision")
        new_revision = discovered["upstream"].get("revision")
        revision_changed = bool(new_revision and new_revision != old_revision)
        for key in ("display_name", "family", "task", "modalities", "parameters"):
            if discovered.get(key):
                output[key] = discovered[key]
        if revision_changed:
            output["upstream"]["revision"] = new_revision
            output["upstream"]["last_checked"] = discovered["upstream"]["last_checked"]
        if discovered["upstream"].get("release_date"):
            output["upstream"]["release_date"] = discovered["upstream"]["release_date"]
        discovered_license = discovered["license"]
        if discovered_license.get("spdx"):
            output["license"] = discovered_license

    for runtime in discovered["compatibility"]["runtimes"]:
        if runtime not in output["compatibility"]["runtimes"]:
            output["compatibility"]["runtimes"].append(runtime)
    output["compatibility"]["runtimes"] = sorted(output["compatibility"]["runtimes"])

    if same_source:
        old_status = output["verification"]["status"]
        new_status = discovered["verification"]["status"]
        if _verification_rank(new_status) > _verification_rank(old_status):
            output["verification"] = discovered["verification"]
    return output


def _semantic_models(document: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(document.get("models", [])))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _source_stale(
    candidate: DiscoveryCandidate,
    now: datetime,
    *,
    max_age_days: int = 365,
) -> bool:
    if not candidate.last_modified:
        return False
    try:
        modified = datetime.fromisoformat(
            candidate.last_modified.replace("Z", "+00:00")
        )
    except ValueError:
        return True
    if modified.tzinfo is None:
        modified = modified.replace(tzinfo=timezone.utc)
    age = now - modified.astimezone(timezone.utc)
    return age > timedelta(days=max_age_days)


@dataclass(frozen=True)
class DiscoveryResult:
    registry: dict[str, Any]
    manifest: dict[str, Any]
    changed: bool
    freshness_refresh: bool
    discovered_count: int
    promoted_count: int


def apply_discovery(
    registry: dict[str, Any],
    candidates: list[DiscoveryCandidate],
    *,
    now: datetime,
    previous_manifest: dict[str, Any] | None = None,
    freshness_days: int = 14,
    expiry_days: int = 90,
) -> DiscoveryResult:
    now = now.astimezone(timezone.utc)
    checked_at = now.isoformat().replace("+00:00", "Z")
    deduped = deduplicate_candidates(candidates)
    output = json.loads(json.dumps(registry))
    existing_by_id, existing_by_url = _existing_index(output)
    used_ids = set(existing_by_id)
    promoted = 0

    for candidate in deduped:
        model_id = _candidate_model_id(candidate, existing_by_url, used_ids)
        converted = candidate_to_registry_model(
            candidate,
            model_id=model_id,
            checked_at=checked_at,
        )
        if converted is None:
            continue
        promoted += 1
        if model_id in existing_by_id:
            merged = _merge_existing(existing_by_id[model_id], converted)
            existing_by_id[model_id].clear()
            existing_by_id[model_id].update(merged)
        else:
            output["models"].append(converted)
            used_ids.add(model_id)
            existing_by_id[model_id] = converted
            existing_by_url[candidate.source_url.rstrip("/").casefold()] = model_id

    output["models"] = sorted(output["models"], key=lambda item: item["id"])
    changed = _semantic_models(output) != _semantic_models(registry)

    generated_raw = registry["registry"]["generated_at"]
    generated_at = datetime.fromisoformat(generated_raw.replace("Z", "+00:00"))
    freshness_refresh = now - generated_at.astimezone(timezone.utc) >= timedelta(
        days=freshness_days
    )
    if changed or freshness_refresh:
        output["registry"]["registry_version"] = (
            int(registry["registry"]["registry_version"]) + 1
        )
        output["registry"]["generated_at"] = checked_at
        output["registry"]["expires_at"] = (
            now + timedelta(days=expiry_days)
        ).isoformat().replace("+00:00", "Z")

    manifest_items: list[dict[str, Any]] = []
    current_keys: set[str] = set()
    for item in deduped:
        value = item.to_manifest_dict()
        value["source_stale"] = _source_stale(item, now)
        current_keys.add(item.identity_key)
        manifest_items.append(value)

    previous_items = (
        previous_manifest.get("candidates", [])
        if isinstance(previous_manifest, dict)
        else []
    )
    for old_item in previous_items:
        if not isinstance(old_item, dict):
            continue
        identity_key = old_item.get("identity_key")
        if not isinstance(identity_key, str) or identity_key in current_keys:
            continue
        tombstone = json.loads(json.dumps(old_item))
        tombstone["deprecated"] = True
        tombstone["lifecycle"] = "DEPRECATED"
        warnings = list(tombstone.get("warnings") or [])
        warnings.append("not observed in latest discovery")
        tombstone["warnings"] = list(dict.fromkeys(warnings))
        manifest_items.append(tombstone)

    manifest_items.sort(
        key=lambda item: (
            str(item.get("identity_key") or ""),
            str(item.get("revision_identity") or ""),
        )
    )
    manifest = {
        "schema_version": 1,
        "generated_at": checked_at,
        "candidate_count": len(deduped),
        "promoted_count": promoted,
        "registry_changed": changed,
        "freshness_refresh": freshness_refresh,
        "candidates": manifest_items,
    }
    return DiscoveryResult(
        registry=output,
        manifest=manifest,
        changed=changed,
        freshness_refresh=freshness_refresh,
        discovered_count=len(deduped),
        promoted_count=promoted,
    )


def run_discovery(
    config: dict[str, Any],
    *,
    now: datetime,
    transport: JsonTransport | None = None,
) -> list[DiscoveryCandidate]:
    transport = transport or JsonTransport()
    trusted = set(config.get("trusted_publishers") or [])
    adapters: list[ModelSourceAdapter] = [
        HuggingFaceAdapter(
            transport=transport,
            trusted_publishers=trusted or None,
            limit_per_task=int(config.get("huggingface_limit_per_task", 25)),
        ),
        NvidiaAdapter(
            transport=transport,
            limit_per_task=int(config.get("nvidia_limit_per_task", 20)),
        ),
        UltralyticsAdapter(transport=transport),
    ]
    vendor_specs = config.get("vendor_github_repositories") or []
    if vendor_specs:
        adapters.append(
            VendorGitHubAdapter(list(vendor_specs), transport=transport)
        )

    records: list[DiscoveryCandidate] = []
    errors: list[str] = []
    for adapter in adapters:
        try:
            records.extend(adapter.discover(now))
        except DiscoveryError as exc:
            errors.append(f"{adapter.name}: {exc}")
    if not records:
        joined = "; ".join(errors) if errors else "no providers returned data"
        raise DiscoveryError(f"discovery produced no candidates: {joined}")
    if errors:
        records.append(
            DiscoveryCandidate(
                provider="vendor-github",
                upstream_id="provider-health",
                source_url="https://github.com/sylvesterkaczmarek/autonomyfit",
                publisher="autonomyfit",
                display_name="Provider health",
                family="provider-health",
                variant=None,
                task=None,
                revision=None,
                release_date=None,
                last_modified=None,
                params_m=None,
                license_id=None,
                library=None,
                runtimes=(),
                evidence_url="https://github.com/sylvesterkaczmarek/autonomyfit",
                ingestion_method="internal-health",
                official_publisher=True,
                metadata_complete=False,
                warnings=tuple(errors),
            )
        )
    return records


def load_discovery_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiscoveryError("discovery config must be a JSON object")
    return value
