from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autonomyfit.discovery import (
    DiscoveryError,
    apply_discovery,
    load_discovery_config,
    run_discovery,
)
from autonomyfit.registry import validate_registry_document


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _manifest_semantic(value: dict[str, Any]) -> str:
    reduced = json.loads(json.dumps(value))
    reduced.pop("generated_at", None)
    reduced.pop("registry_changed", None)
    reduced.pop("freshness_refresh", None)
    for item in reduced.get("candidates", []):
        if isinstance(item, dict):
            item.pop("downloads", None)
            item.pop("likes", None)
    return json.dumps(reduced, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover upstream models and update the AutonomyFit registry."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("registry/source/registry-v2.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("registry/discovery/config.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("registry/discovery/latest.json"),
    )
    parser.add_argument(
        "--now",
        help="UTC ISO timestamp override used by deterministic tests.",
    )
    args = parser.parse_args()

    now = (
        datetime.fromisoformat(args.now.replace("Z", "+00:00"))
        if args.now
        else datetime.now(timezone.utc)
    )
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    registry = _read_json(args.registry)
    config = load_discovery_config(args.config)
    try:
        candidates = run_discovery(config, now=now)
    except DiscoveryError as exc:
        raise SystemExit(f"discovery failed closed: {exc}") from exc

    old_manifest = (
        _read_json(args.manifest)
        if args.manifest.exists()
        else {}
    )
    result = apply_discovery(
        registry,
        candidates,
        now=now,
        previous_manifest=old_manifest,
    )
    validate_registry_document(result.registry)

    manifest_changed = (
        _manifest_semantic(old_manifest)
        != _manifest_semantic(result.manifest)
    )
    should_write = (
        result.changed
        or result.freshness_refresh
        or manifest_changed
    )

    if should_write:
        if result.changed or result.freshness_refresh:
            _write_json(args.registry, result.registry)
        _write_json(args.manifest, result.manifest)

    summary = {
        "candidate_count": result.discovered_count,
        "promoted_count": result.promoted_count,
        "registry_changed": result.changed,
        "freshness_refresh": result.freshness_refresh,
        "manifest_changed": manifest_changed,
        "files_changed": should_write,
        "registry_version": (
            result.registry["registry"]["registry_version"]
            if result.changed or result.freshness_refresh
            else registry["registry"]["registry_version"]
        ),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
