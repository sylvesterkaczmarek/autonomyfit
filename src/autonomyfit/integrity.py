from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_members(path: Path) -> tuple[Path, ...]:
    """Return the byte-bearing members that define one deployment artifact identity."""
    path = path.expanduser().resolve()
    if path.is_dir():
        members = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
        if not members:
            raise ValueError(f"artifact directory is empty: {path}")
        return members
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {path}")
    if path.suffix.casefold() == ".xml":
        companion = path.with_suffix(".bin")
        if companion.is_file():
            return (path, companion)
    return (path,)


def artifact_sha256(path: Path) -> str:
    """SHA-256 for one file, or a deterministic manifest digest for a bundle/directory."""
    path = path.expanduser().resolve()
    members = artifact_members(path)
    if len(members) == 1 and members[0] == path and path.is_file():
        return sha256_file(path)
    root = path if path.is_dir() else path.parent
    digest = hashlib.sha256()
    for member in members:
        relative = member.relative_to(root).as_posix().encode("utf-8")
        file_digest = sha256_file(member).encode("ascii")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest)
    return digest.hexdigest()


def artifact_size_bytes(path: Path) -> int:
    return sum(member.stat().st_size for member in artifact_members(path))
