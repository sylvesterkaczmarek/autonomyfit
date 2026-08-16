from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autonomyfit.registry import models_from_registry, validate_registry_document  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "registry/source/registry-v2.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_registry_document(document)
    models = models_from_registry(document)
    version = document["registry"]["registry_version"]
    print(f"registry v{version}: {len(models)} models valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
