from __future__ import annotations

import argparse
from pathlib import Path

from autonomyfit.catalog import load_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path, nargs="?")
    args = parser.parse_args()
    models = load_models(args.catalog)
    print(f"valid: {len(models)} model profiles")


if __name__ == "__main__":
    main()
