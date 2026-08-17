from __future__ import annotations

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
            value = stripped.split("uses:", 1)[1].strip().split(" #", 1)[0].strip().strip("\"'")
            if value.startswith("./"):
                continue
            if not re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", value):
                failures.append(f"{path}:{line_number}: {value}")
    assert not failures, "mutable or unpinned workflow actions: " + "; ".join(failures)
