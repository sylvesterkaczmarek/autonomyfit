from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import artifact_format
from .integrity import artifact_sha256, sha256_file


class ConversionError(RuntimeError):
    """Artifact conversion failed or is not safely automatable."""


@dataclass(frozen=True)
class ConversionResult:
    source_path: Path
    source_sha256: str
    source_format: str
    target_path: Path
    target_sha256: str
    target_format: str
    target_runtime: str
    tool: str
    tool_version: str | None
    command: tuple[str, ...]
    duration_s: float
    built_locally: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)
    equivalence: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "not-run",
            "reason": "conversion success does not establish numerical or task-level equivalence",
        }
    )
    companion_artifacts: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_path"] = str(self.source_path)
        value["target_path"] = str(self.target_path)
        return value


def _tool_version(command: list[str], timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0][:300] if text else None


def _run(command: list[str], *, timeout: float = 1200.0) -> tuple[float, str]:
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConversionError(f"conversion command could not run: {exc}") from exc
    duration = time.monotonic() - started
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise ConversionError(
            f"conversion failed with exit code {result.returncode}: {output[-1800:]}"
        )
    return duration, output


def _result(
    source: Path,
    target: Path,
    runtime: str,
    *,
    tool: str,
    tool_version: str | None,
    command: list[str],
    duration_s: float,
    warnings: list[str] | None = None,
    companion_paths: list[Path] | None = None,
) -> ConversionResult:
    if not target.exists():
        raise ConversionError(f"conversion reported success but output is missing: {target}")
    companions = []
    for path in companion_paths or []:
        if path.exists() and path.is_file():
            companions.append(
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            )
    return ConversionResult(
        source_path=source,
        source_sha256=artifact_sha256(source),
        source_format=artifact_format(source),
        target_path=target,
        target_sha256=artifact_sha256(target),
        target_format=artifact_format(target),
        target_runtime=runtime,
        tool=tool,
        tool_version=tool_version,
        command=tuple(command),
        duration_s=round(duration_s, 3),
        warnings=tuple(warnings or []),
        companion_artifacts=tuple(companions),
    )


def convert_to_tensorrt(
    source: Path,
    output_dir: Path,
    *,
    precision: str = "fp16",
    input_shapes: dict[str, list[int]] | None = None,
) -> ConversionResult:
    if source.suffix.casefold() != ".onnx":
        raise ConversionError("TensorRT conversion currently requires an ONNX source artifact")
    executable = shutil.which("trtexec")
    if not executable:
        raise ConversionError("TensorRT conversion requires trtexec on PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{source.stem}.{precision.casefold()}.engine"
    command = [executable, f"--onnx={source}", f"--saveEngine={target}", "--skipInference"]
    normalized = precision.casefold()
    if normalized == "fp16":
        command.append("--fp16")
    elif normalized == "int8":
        command.append("--int8")
    elif normalized not in {"fp32", "artifact"}:
        raise ConversionError(f"unsupported TensorRT conversion precision: {precision}")
    if input_shapes:
        shape_arg = ",".join(
            f"{name}:{'x'.join(str(dim) for dim in dims)}"
            for name, dims in sorted(input_shapes.items())
        )
        command.append(f"--shapes={shape_arg}")
    duration, _ = _run(command)
    return _result(
        source,
        target,
        "tensorrt",
        tool="trtexec",
        tool_version=_tool_version([executable, "--version"]),
        command=command,
        duration_s=duration,
        warnings=[
            "TensorRT engines are machine/toolchain-sensitive executable artifacts; keep the locally built plan bound to its recorded stack."
        ],
    )


def convert_to_openvino(
    source: Path,
    output_dir: Path,
    *,
    precision: str = "fp32",
) -> ConversionResult:
    if source.suffix.casefold() != ".onnx":
        raise ConversionError("OpenVINO conversion currently requires an ONNX source artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{source.stem}.xml"
    executable = shutil.which("ovc")
    if executable:
        command = [executable, str(source), "--output_model", str(target)]
        duration, _ = _run(command)
        bin_path = target.with_suffix(".bin")
        return _result(
            source,
            target,
            "openvino",
            tool="ovc",
            tool_version=_tool_version([executable, "--version"]),
            command=command,
            duration_s=duration,
            companion_paths=[bin_path],
        )
    try:
        import openvino as ov
    except ImportError as exc:
        raise ConversionError(
            "OpenVINO conversion requires openvino/ovc; install the OpenVINO Python package"
        ) from exc
    started = time.monotonic()
    try:
        model = ov.convert_model(str(source))
        ov.save_model(model, str(target), compress_to_fp16=precision.casefold() == "fp16")
    except Exception as exc:
        raise ConversionError(f"OpenVINO conversion failed: {exc}") from exc
    duration = time.monotonic() - started
    version = getattr(ov, "__version__", None)
    command = ["python:openvino.convert_model", str(source), str(target)]
    return _result(
        source,
        target,
        "openvino",
        tool="openvino.convert_model",
        tool_version=version,
        command=command,
        duration_s=duration,
        companion_paths=[target.with_suffix(".bin")],
    )


def convert_trusted_torchscript(
    source: Path,
    target_runtime: str,
    output_dir: Path,
    *,
    input_shape: list[int] | None,
    precision: str = "fp32",
    trust_source: bool = False,
) -> ConversionResult:
    if source.suffix.casefold() not in {".pt", ".pth"}:
        raise ConversionError("trusted PyTorch conversion requires a local .pt or .pth TorchScript artifact")
    if not trust_source:
        raise ConversionError(
            "loading PyTorch/TorchScript serialization is not an automatic safe operation; re-run with explicit trust only for an artifact you control"
        )
    if not input_shape:
        raise ConversionError("trusted PyTorch conversion requires an explicit --shape")
    try:
        import torch
    except ImportError as exc:
        raise ConversionError("PyTorch conversion requires torch") from exc
    try:
        module = torch.jit.load(str(source), map_location="cpu")
        module.eval()
    except Exception as exc:
        raise ConversionError(f"could not load trusted TorchScript artifact: {exc}") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(tuple(input_shape), dtype=torch.float32)
    normalized = target_runtime.casefold()
    if normalized in {"onnx", "onnxruntime"}:
        target = output_dir / f"{source.stem}.onnx"
        started = time.monotonic()
        try:
            torch.onnx.export(module, (example,), str(target), dynamo=True)
        except Exception as exc:
            raise ConversionError(f"PyTorch to ONNX export failed: {exc}") from exc
        duration = time.monotonic() - started
        return _result(
            source,
            target,
            "onnxruntime",
            tool="torch.onnx.export",
            tool_version=getattr(torch, "__version__", None),
            command=["python:torch.onnx.export", str(source), str(target)],
            duration_s=duration,
            warnings=["Source deserialization was explicitly trusted by the user."],
        )
    if normalized == "coreml":
        try:
            import coremltools as ct
        except ImportError as exc:
            raise ConversionError("Core ML conversion requires coremltools") from exc
        target = output_dir / f"{source.stem}.mlpackage"
        started = time.monotonic()
        try:
            converted = ct.convert(
                module,
                inputs=[ct.TensorType(shape=tuple(input_shape))],
                convert_to="mlprogram",
                compute_precision=(ct.precision.FLOAT16 if precision.casefold() == "fp16" else ct.precision.FLOAT32),
            )
            converted.save(str(target))
        except Exception as exc:
            raise ConversionError(f"PyTorch to Core ML conversion failed: {exc}") from exc
        duration = time.monotonic() - started
        return ConversionResult(
            source_path=source,
            source_sha256=artifact_sha256(source),
            source_format=artifact_format(source),
            target_path=target,
            target_sha256=artifact_sha256(target),
            target_format="coreml-package",
            target_runtime="coreml",
            tool="coremltools.convert",
            tool_version=getattr(ct, "__version__", None),
            command=("python:coremltools.convert", str(source), str(target)),
            duration_s=round(duration, 3),
            built_locally=True,
            warnings=(
                "Source deserialization was explicitly trusted by the user.",
                "Core ML conversion success does not establish task-level accuracy equivalence.",
            ),
        )
    raise ConversionError(f"unsupported trusted PyTorch target runtime: {target_runtime}")


def convert_artifact(
    source: Path,
    target_runtime: str,
    output_dir: Path,
    *,
    precision: str = "fp16",
    input_shape: list[int] | None = None,
    input_shapes: dict[str, list[int]] | None = None,
    trust_source: bool = False,
) -> ConversionResult:
    normalized = target_runtime.strip().casefold()
    fmt = artifact_format(source)
    if normalized in {"onnx", "onnxruntime"} and fmt == "onnx":
        return ConversionResult(
            source_path=source,
            source_sha256=artifact_sha256(source),
            source_format=fmt,
            target_path=source,
            target_sha256=artifact_sha256(source),
            target_format=fmt,
            target_runtime="onnxruntime",
            tool="identity",
            tool_version=None,
            command=(),
            duration_s=0.0,
            built_locally=False,
            warnings=("No conversion was required.",),
        )
    if normalized == "tensorrt":
        return convert_to_tensorrt(source, output_dir, precision=precision, input_shapes=input_shapes)
    if normalized == "openvino":
        return convert_to_openvino(source, output_dir, precision=precision)
    if fmt == "pytorch":
        return convert_trusted_torchscript(
            source,
            normalized,
            output_dir,
            input_shape=input_shape,
            precision=precision,
            trust_source=trust_source,
        )
    if normalized == "coreml":
        raise ConversionError(
            "generic ONNX-to-Core ML conversion is intentionally not automated; current Core ML Tools guidance prefers direct PyTorch/TensorFlow conversion, which requires a trusted source graph and input contract"
        )
    raise ConversionError(f"no safe conversion path from {fmt} to {target_runtime}")


def compare_onnx_openvino_outputs(
    source_onnx: Path,
    target_xml: Path,
    *,
    shape_override: list[int] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Compare deterministic numeric outputs when the two runtimes expose a compatible contract."""
    try:
        import numpy as np
        import onnxruntime as ort
        import openvino as ov
    except ImportError as exc:
        return {"status": "unsupported", "reason": f"equivalence dependencies unavailable: {exc}"}
    try:
        session = ort.InferenceSession(str(source_onnx), providers=["CPUExecutionProvider"])
        ort_inputs = session.get_inputs()
        if shape_override is not None and len(ort_inputs) != 1:
            return {"status": "unsupported", "reason": "shape override is generic only for a single input"}
        feeds: dict[str, Any] = {}
        rng = np.random.default_rng(0)
        for meta in ort_inputs:
            dims: list[int] = []
            for index, dim in enumerate(meta.shape):
                if shape_override is not None:
                    dims = list(shape_override)
                    break
                if isinstance(dim, int) and dim > 0:
                    dims.append(dim)
                elif index == 0:
                    dims.append(1)
                else:
                    return {"status": "unsupported", "reason": f"dynamic input {meta.name!r} needs an explicit shape"}
            if meta.type not in {"tensor(float)", "tensor(float16)"}:
                return {"status": "unsupported", "reason": f"input type {meta.type} is not covered by the generic numeric check"}
            dtype = np.float16 if meta.type == "tensor(float16)" else np.float32
            feeds[meta.name] = rng.random(dims, dtype=np.float32).astype(dtype)
        reference = session.run(None, feeds)

        core = ov.Core()
        compiled = core.compile_model(str(target_xml), "CPU")
        ov_feeds: dict[Any, Any] = {}
        by_name = {item.any_name: item for item in compiled.inputs}
        for name, value in feeds.items():
            port = by_name.get(name)
            if port is None:
                return {"status": "unsupported", "reason": f"converted input name {name!r} was not preserved"}
            ov_feeds[port] = value
        result_map = compiled(ov_feeds)
        converted = [result_map[port] for port in compiled.outputs]
        if len(reference) != len(converted):
            return {"status": "failed", "reason": "output count differs after conversion"}
        max_abs = 0.0
        for ref, got in zip(reference, converted):
            if ref.shape != got.shape:
                return {"status": "failed", "reason": f"output shape differs: {ref.shape} != {got.shape}"}
            if not np.issubdtype(ref.dtype, np.number) or not np.issubdtype(got.dtype, np.number):
                return {"status": "unsupported", "reason": "non-numeric output cannot be checked generically"}
            if ref.size:
                max_abs = max(max_abs, float(np.max(np.abs(ref.astype(np.float64) - got.astype(np.float64)))))
            if not np.allclose(ref, got, rtol=rtol, atol=atol, equal_nan=True):
                return {
                    "status": "failed",
                    "reason": "numeric outputs exceed configured tolerance",
                    "rtol": rtol,
                    "atol": atol,
                    "max_abs_error": max_abs,
                }
        return {
            "status": "passed",
            "reason": "deterministic synthetic numeric outputs matched within tolerance; task accuracy was not evaluated",
            "rtol": rtol,
            "atol": atol,
            "max_abs_error": max_abs,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "unsupported", "reason": f"generic equivalence check could not run: {exc}"}
