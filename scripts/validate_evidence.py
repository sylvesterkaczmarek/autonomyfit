from __future__ import annotations

from autonomyfit.evidence import load_evidence_store


def main() -> int:
    store = load_evidence_store(include_local=False)
    eligible = sum(item.eligible_for_verified_fit for item in store.benchmarks)
    print(f"evidence v2: {len(store.benchmarks)} benchmark records valid; {eligible} bundled records eligible for VERIFIED_FIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
