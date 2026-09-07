from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath


def onnx_external_locations(path: Path) -> tuple[str, ...]:
    """Inspect references without opening external data or executing the graph."""
    try:
        import onnx
    except ImportError as exc:
        raise ValueError(
            "ONNX artifact identity requires its parser; install 'autonomyfit[deployment]'"
        ) from exc
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise ValueError(f"could not inspect ONNX artifact: {exc}") from exc

    def tensors(message):
        # Reflection includes constants, sparse tensors, nested graphs and functions,
        # as well as the graph's top-level initializers.
        if isinstance(message, onnx.TensorProto):
            yield message
            return
        for descriptor, value in message.ListFields():
            if descriptor.type != descriptor.TYPE_MESSAGE:
                continue
            repeated = getattr(descriptor, "is_repeated", None)
            if repeated is None:  # Older protobuf versions expose only label.
                repeated = descriptor.label == descriptor.LABEL_REPEATED
            if repeated:
                for item in value:
                    yield from tensors(item)
            else:
                yield from tensors(value)

    locations: set[str] = set()
    for tensor in tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.external_data:
            continue
        entries = {entry.key: entry.value for entry in tensor.external_data}
        if len(entries) != len(tensor.external_data):
            raise ValueError("ONNX external tensor has duplicate metadata keys")
        location = entries.get("location", "")
        relative = PurePosixPath(location)
        if (
            not location
            or "\x00" in location
            or "\\" in location
            or relative.is_absolute()
            or PureWindowsPath(location).drive
            or ".." in relative.parts
            or not relative.parts
        ):
            raise ValueError(f"unsafe ONNX external tensor path: {location!r}")
        locations.add(relative.as_posix())
    return tuple(sorted(locations))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_members(path: Path) -> tuple[Path, ...]:
    """Return the byte-bearing members that define one deployment artifact identity."""
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"artifact root may not be a symbolic link: {expanded}")
    path = expanded.resolve()
    if path.is_dir():
        entries = tuple(sorted(path.rglob("*")))
        symlinks = [item for item in entries if item.is_symlink()]
        if symlinks:
            raise ValueError(
                "artifact bundle may not contain symbolic links: "
                + ", ".join(str(item.relative_to(path)) for item in symlinks[:5])
            )
        members = tuple(item for item in entries if item.is_file())
        if not members:
            raise ValueError(f"artifact directory is empty: {path}")
        return members
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {path}")
    if path.suffix.casefold() == ".onnx":
        companions = []
        for location in onnx_external_locations(path):
            companion = path.parent / location
            current = path.parent
            for component in PurePosixPath(location).parts:
                current = current / component
                if current.is_symlink():
                    raise ValueError(f"ONNX companion may not contain symbolic links: {companion}")
            if not companion.is_file():
                raise ValueError(f"ONNX external tensor file is missing: {companion}")
            if companion == path:
                raise ValueError("ONNX external tensor data may not reference its own graph")
            companions.append(companion)
        return (path, *companions)
    if path.suffix.casefold() == ".xml":
        companion = path.with_suffix(".bin")
        if companion.is_symlink():
            raise ValueError(f"OpenVINO companion may not be a symbolic link: {companion}")
        if companion.is_file():
            return (path, companion)
    return (path,)


def artifact_sha256(path: Path) -> str:
    """SHA-256 for one file, or a deterministic manifest digest for a bundle/directory."""
    path = path.expanduser()
    members = artifact_members(path)
    path = path.resolve()
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
