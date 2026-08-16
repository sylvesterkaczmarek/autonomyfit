from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .backends import BackendError, BenchmarkRequest, backend_status, run_benchmark
from .benchmark import save_result
from .catalog import load_hardware_profiles, load_model_catalog
from .evidence import (
    EvidenceError,
    import_benchmark_report,
    inspect_benchmark_report,
    validate_benchmark_report,
)
from .hardware import detect_hardware, hardware_from_profile
from .models import Constraints, ModelProfile
from .registry import RegistryClient, RegistryError
from .reporting import print_hardware, print_recommendations
from .scoring import recommend_models

app = typer.Typer(
    name="autonomyfit",
    help="Find edge-AI models that fit hardware and deployment constraints.",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)
registry_app = typer.Typer(help="Inspect and manage the signed model registry.")
app.add_typer(registry_app, name="registry")
console = Console()


def _resolve_hardware(profile: str | None):
    return hardware_from_profile(profile) if profile else detect_hardware()


def _parse_shape(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        dims = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise typer.BadParameter(
            "shape must be comma-separated positive integers, for example 1,3,640,640",
            param_hint="--shape",
        ) from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise typer.BadParameter("shape dimensions must be positive integers", param_hint="--shape")
    return dims


def _print_registry_status(status: dict[str, object], as_json: bool) -> None:
    if as_json:
        console.print_json(json.dumps(status))
        return
    table = Table(title="AutonomyFit registry", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    cache = status.get("cache")
    fallback = status["fallback"]
    table.add_row("Official registry", str(status["registry_url"]))
    table.add_row("Cache directory", str(status["cache_dir"]))
    if isinstance(cache, dict):
        table.add_row("Cached version", str(cache.get("registry_version")))
        table.add_row("Cached at", str(cache.get("cached_at") or "unknown"))
        table.add_row("Cached expires", str(cache.get("expires_at") or "unknown"))
        table.add_row("Cached stale", "yes" if cache.get("stale") else "no")
    else:
        table.add_row("Cache", "empty")
    if isinstance(fallback, dict):
        table.add_row("Fallback version", str(fallback.get("registry_version")))
    table.add_row("Highest trusted version", str(status.get("highest_seen_version") or "none"))
    console.print(table)


def _model_payload(model: ModelProfile) -> dict[str, object]:
    return asdict(model)


def _load_filtered_models(
    *, task: str | None = None, source: str | None = None,
    status: str | None = None, new_since: date | None = None,
    offline: bool = False,
) -> tuple[list[ModelProfile], object]:
    loaded = load_model_catalog(offline=offline)
    models = list(loaded.models)
    if task:
        normalized_task = task.strip().lower()
        if normalized_task not in {"detection", "vlm"}:
            raise typer.BadParameter("task must be detection or vlm", param_hint="--task")
        models = [model for model in models if model.task == normalized_task]
    if source:
        needle = source.casefold()
        models = [
            model for model in models
            if needle in model.source_id.casefold() or needle in model.source_url.casefold()
        ]
    if status:
        normalized_status = status.strip().lower()
        models = [model for model in models if model.verification_status.casefold() == normalized_status]
    if new_since:
        models = [
            model for model in models
            if model.release_date and date.fromisoformat(model.release_date) >= new_since
        ]
    return models, loaded.provenance


def _models_table(models: list[ModelProfile], title: str) -> Table:
    table = Table(title=title)
    table.add_column("ID")
    table.add_column("Task")
    table.add_column("Parameters")
    table.add_column("Licence")
    table.add_column("Source")
    table.add_column("Verification")
    for model in models:
        table.add_row(
            model.id, model.task, f"{model.params_m:g}M",
            model.license_spdx or "unknown", model.source_id, model.verification_status,
        )
    return table


def _resolve_model_revision(model_id: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    try:
        loaded = load_model_catalog(offline=True)
    except (OSError, ValueError, RegistryError):
        return None
    needle = model_id.casefold()
    model = next((item for item in loaded.models if item.id.casefold() == needle), None)
    return model.source_revision if model else None


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        from importlib.metadata import version as package_version
        console.print(package_version("autonomyfit"))
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        hardware = detect_hardware()
        print_hardware(hardware)
        console.print()
        print_recommendations(recommend_models(hardware, Constraints(task="detection")), limit=5)


@app.command()
def scan(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Detect compute, memory, accelerator and runtime readiness."""
    print_hardware(detect_hardware(), as_json=json_output)


@app.command()
def recommend(
    task: Annotated[str, typer.Option(help="Model task: detection or vlm.")] = "detection",
    fps: Annotated[float | None, typer.Option("--fps", help="Minimum required FPS.", min=0.0)] = None,
    latency_ms: Annotated[
        float | None,
        typer.Option("--latency-ms", help="Maximum inference latency in milliseconds.", min=0.0),
    ] = None,
    power_w: Annotated[
        float | None, typer.Option("--power-w", help="Maximum measured power in watts.", min=0.0)
    ] = None,
    min_accuracy: Annotated[
        float | None,
        typer.Option("--min-accuracy", help="Minimum catalog accuracy value.", min=0.0),
    ] = None,
    runtime: Annotated[str | None, typer.Option(help="Force target runtime.")] = None,
    precision: Annotated[
        str | None, typer.Option(help="Force precision, for example fp16 or int8.")
    ] = None,
    model_id: Annotated[
        str | None, typer.Option("--model-id", help="Restrict to one canonical model ID.")
    ] = None,
    model_revision: Annotated[
        str | None, typer.Option("--model-revision", help="Pin an exact upstream model revision.")
    ] = None,
    artifact: Annotated[
        Path | None,
        typer.Option("--artifact", help="Exact local artifact whose SHA-256 should match benchmark evidence."),
    ] = None,
    hardware_profile: Annotated[
        str | None,
        typer.Option("--hardware-profile", help="Use a bundled hardware profile instead of local detection."),
    ] = None,
    catalog: Annotated[
        Path | None, typer.Option(help="Optional custom model catalog JSON (schema v1 or v2).")
    ] = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Use verified cache or bundled fallback only.")
    ] = False,
    limit: Annotated[int, typer.Option(help="Maximum number of candidates to show.", min=1)] = 8,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Rank models using evidence that is explicit about identity and measurement scope."""
    task = task.strip().lower()
    if task not in {"detection", "vlm"}:
        raise typer.BadParameter("task must be detection or vlm", param_hint="--task")
    if artifact is not None:
        if not artifact.exists():
            raise typer.BadParameter(f"artifact does not exist: {artifact}", param_hint="--artifact")
        if not model_id:
            raise typer.BadParameter("--artifact requires --model-id", param_hint="--artifact")
    try:
        hardware = _resolve_hardware(hardware_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--hardware-profile") from exc
    constraints = Constraints(
        task=task,  # type: ignore[arg-type]
        min_fps=fps,
        max_latency_ms=latency_ms,
        max_power_w=power_w,
        min_accuracy=min_accuracy,
        runtime=runtime.strip().lower() if runtime else None,
        precision=precision.strip().lower() if precision else None,
        model_id=model_id,
        model_revision=model_revision,
        artifact_path=artifact,
    )
    try:
        items = recommend_models(hardware, constraints, catalog_path=catalog, offline=offline)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RegistryError, EvidenceError) as exc:
        if catalog is None:
            raise typer.BadParameter(f"evidence/registry error: {exc}") from exc
        raise typer.BadParameter(f"invalid catalog: {exc}", param_hint="--catalog") from exc
    print_recommendations(items, limit=limit, as_json=json_output)


@app.command("models")
def models_command(
    task: Annotated[str | None, typer.Option(help="Filter to detection or vlm.")] = None,
    source: Annotated[str | None, typer.Option(help="Filter by source or publisher identifier.")] = None,
    status: Annotated[str | None, typer.Option(help="Filter by verification state.")] = None,
    new_since: Annotated[
        str | None, typer.Option("--new-since", help="Only show models released on/after YYYY-MM-DD.")
    ] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Use verified cache/fallback only.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """List models from the signed registry with discovery filters."""
    parsed_since = None
    if new_since:
        try:
            parsed_since = date.fromisoformat(new_since)
        except ValueError as exc:
            raise typer.BadParameter("new-since must be YYYY-MM-DD", param_hint="--new-since") from exc
    models, provenance = _load_filtered_models(
        task=task, source=source, status=status, new_since=parsed_since, offline=offline
    )
    if json_output:
        console.print_json(json.dumps({
            "registry": provenance.to_dict(), "models": [_model_payload(model) for model in models]
        }))
        return
    console.print(_models_table(models, "Model registry"))
    console.print(f"Registry v{provenance.registry_version or '?'} · {provenance.source}")
    if provenance.warning:
        console.print(f"Warning: {provenance.warning}")


@app.command("search")
def search_command(
    query: Annotated[str, typer.Argument(help="Search terms.")],
    task: Annotated[str | None, typer.Option(help="Optional task filter.")] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Use verified cache/fallback only.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Search model identity, family and upstream source metadata."""
    models, provenance = _load_filtered_models(task=task, offline=offline)
    terms = [term.casefold() for term in query.split() if term.strip()]
    matches=[]
    for model in models:
        haystack = " ".join([
            model.id, model.display_name, model.family, model.variant or "", model.task,
            " ".join(model.input_modalities), " ".join(model.output_modalities),
            model.source_id, model.source_url, model.license_spdx or "",
        ]).casefold()
        if all(term in haystack for term in terms):
            matches.append(model)
    if json_output:
        console.print_json(json.dumps({
            "query": query, "registry": provenance.to_dict(),
            "models": [_model_payload(model) for model in matches],
        }))
    else:
        console.print(_models_table(matches, f"Search: {query}"))


@app.command("info")
def info_command(
    model_id: Annotated[str, typer.Argument(help="Canonical model ID.")],
    offline: Annotated[bool, typer.Option("--offline", help="Use verified cache/fallback only.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show normalized model metadata and provenance."""
    loaded=load_model_catalog(offline=offline)
    needle=model_id.casefold()
    model=next((item for item in loaded.models if item.id.casefold()==needle or item.display_name.casefold()==needle),None)
    if model is None:
        raise typer.BadParameter(f"unknown model: {model_id}")
    if json_output:
        console.print_json(json.dumps({"registry": loaded.provenance.to_dict(), "model": _model_payload(model)}))
        return
    table=Table(title=model.display_name, show_header=False)
    table.add_column("Field", style="bold"); table.add_column("Value")
    rows=[
        ("ID",model.id),("Family",model.family),("Variant",model.variant or "-"),("Task",model.task),
        ("Parameters",f"{model.params_m:g}M"),("Source",model.source_id),("Source URL",model.source_url),
        ("Revision",model.source_revision or "unknown"),("Released",model.release_date or "unknown"),
        ("Last checked",model.last_checked or "unknown"),("Runtimes",", ".join(model.runtimes)),
        ("Licence",model.license_spdx or "unknown"),("Licence status",model.license_status),
        ("Verification",model.verification_status),("Last verified",model.last_verified or "unknown"),
    ]
    for key,value in rows: table.add_row(key,value)
    console.print(table)


@app.command("catalog")
def catalog_command(
    task: Annotated[str | None, typer.Option(help="Filter to detection or vlm.")] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Use verified cache/fallback only.")] = False,
) -> None:
    """Backward-compatible alias for listing the model registry."""
    models, provenance=_load_filtered_models(task=task,offline=offline)
    console.print(_models_table(models,"Model registry"))
    console.print(f"Registry v{provenance.registry_version or '?'} · {provenance.source}")


@app.command("profiles")
def profiles_command() -> None:
    """List bundled hardware profiles with published evidence coverage."""
    profiles=load_hardware_profiles()
    table=Table(title="Hardware profiles")
    table.add_column("ID"); table.add_column("Hardware"); table.add_column("Memory"); table.add_column("Platform")
    for profile_id,item in sorted(profiles.items()):
        table.add_row(profile_id,item["display_name"],f"{item['memory_gb']} GB",item["platform"])
    console.print(table)


@registry_app.command("status")
def registry_status(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show local registry cache and trust state without network access."""
    _print_registry_status(RegistryClient().status(),json_output)


@registry_app.command("update")
def registry_update(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Fetch and verify the latest official registry now."""
    try:
        snapshot=RegistryClient().update()
    except RegistryError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload=snapshot.provenance.to_dict(); payload["model_count"]=len(snapshot.models)
    if json_output: console.print_json(json.dumps(payload))
    else: console.print(f"Updated to registry v{snapshot.provenance.registry_version} ({len(snapshot.models)} models), Sigstore verified.")


@registry_app.command("clear-cache")
def registry_clear_cache(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Remove cached registry data while preserving rollback protection."""
    removed=RegistryClient().clear_cache()
    payload={"removed":removed,"security_state_preserved":True}
    if json_output: console.print_json(json.dumps(payload))
    else: console.print(f"Removed {len(removed)} cache file(s). Rollback trust state preserved.")


@app.command()
def benchmark(
    model: Annotated[Path, typer.Argument(help="Path to a local model artifact.")],
    model_id: Annotated[str | None, typer.Option("--model-id", help="Canonical model identity.")] = None,
    model_revision: Annotated[str | None, typer.Option("--model-revision", help="Exact upstream revision.")] = None,
    backend: Annotated[str | None, typer.Option(help="onnxruntime, tensorrt, openvino or coreml.")] = None,
    iterations: Annotated[int, typer.Option(help="Timed inference iterations.", min=1)] = 50,
    warmup: Annotated[int, typer.Option(help="Warm-up setting for the selected backend.", min=0)] = 10,
    shape: Annotated[str | None, typer.Option(help="ONNX dynamic input shape, e.g. 1,3,640,640.")] = None,
    provider: Annotated[str | None, typer.Option(help="Backend provider override.")] = None,
    device: Annotated[str | None, typer.Option(help="Native backend device, e.g. CPU, GPU, NPU.")] = None,
    precision: Annotated[str, typer.Option(help="Artifact/runtime precision label.")] = "artifact",
    quantization: Annotated[str | None, typer.Option(help="Quantization label, e.g. int8.")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", min=1)] = None,
    output: Annotated[Path | None, typer.Option("--output","-o",help="Write benchmark report JSON.")] = None,
) -> None:
    """Benchmark an exact local artifact and emit a schema-v2 evidence report."""
    if not model.exists() or not model.is_file():
        raise typer.BadParameter(f"model artifact does not exist: {model}")
    shape_override=_parse_shape(shape)
    resolved_id=model_id or model.stem
    resolved_revision=_resolve_model_revision(resolved_id,model_revision)
    hardware=detect_hardware()
    request=BenchmarkRequest(
        model_path=model,model_id=resolved_id,model_revision=resolved_revision,
        hardware=hardware,iterations=iterations,warmup=warmup,shape_override=shape_override,
        provider=provider,device=device,precision=precision.strip().lower(),
        quantization=quantization.strip().lower() if quantization else None,batch_size=batch_size,
    )
    try:
        report=run_benchmark(request,backend)
        validate_benchmark_report(report)
    except (BackendError,EvidenceError,ValueError,RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(report))
    if output:
        save_result(report,output); console.print(f"Saved {output}")


@app.command("benchmark-backends")
def benchmark_backends_command(
    json_output: Annotated[bool,typer.Option("--json",help="Emit machine-readable JSON.")]=False,
) -> None:
    """Show availability of native benchmark backends."""
    statuses=backend_status()
    payload=[asdict(item) for item in statuses]
    if json_output: console.print_json(json.dumps(payload)); return
    table=Table(title="Benchmark backends")
    table.add_column("Backend"); table.add_column("Available"); table.add_column("Version"); table.add_column("Detail")
    for item in statuses:
        table.add_row(item.name,"yes" if item.available else "no",item.version or "-",item.detail or item.executable or "-")
    console.print(table)


@app.command("benchmark-inspect")
def benchmark_inspect_command(
    report: Annotated[Path,typer.Argument(help="AutonomyFit benchmark report JSON.")],
) -> None:
    """Validate and inspect a benchmark report without importing it."""
    try:
        payload=inspect_benchmark_report(report)
    except EvidenceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(payload))


@app.command("benchmark-import")
def benchmark_import_command(
    report: Annotated[Path,typer.Argument(help="AutonomyFit benchmark report JSON.")],
) -> None:
    """Validate and import a local benchmark into the user evidence store."""
    try:
        target=import_benchmark_report(report)
        payload=inspect_benchmark_report(target)
    except EvidenceError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload["imported_path"]=str(target)
    console.print_json(json.dumps(payload))


if __name__ == "__main__":
    app()
