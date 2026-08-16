from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .benchmark import benchmark_onnx, save_result
from .catalog import load_hardware_profiles, load_model_catalog
from .hardware import detect_hardware, hardware_from_profile
from .models import Constraints
from .registry import RegistryClient, RegistryError
from .reporting import print_hardware, print_recommendations
from .scoring import recommend_models

app = typer.Typer(
    name="autonomyfit",
    help="Find edge-AI models that fit your hardware and deployment constraints.",
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
        items = recommend_models(hardware, Constraints(task="detection"))
        print_recommendations(items, limit=5)


@app.command()
def scan(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Detect compute, memory, accelerator and runtime readiness."""
    print_hardware(detect_hardware(), as_json=json_output)


@app.command()
def recommend(
    task: Annotated[str, typer.Option(help="Model task: detection or vlm.")] = "detection",
    fps: Annotated[
        float | None, typer.Option("--fps", help="Minimum required FPS.", min=0.0)
    ] = None,
    latency_ms: Annotated[
        float | None,
        typer.Option("--latency-ms", help="Maximum inference latency in milliseconds.", min=0.0),
    ] = None,
    power_w: Annotated[
        float | None,
        typer.Option("--power-w", help="Maximum measured power in watts.", min=0.0),
    ] = None,
    min_accuracy: Annotated[
        float | None,
        typer.Option(
            "--min-accuracy",
            help="Minimum catalog accuracy value for the task metric.",
            min=0.0,
        ),
    ] = None,
    runtime: Annotated[str | None, typer.Option(help="Force target runtime.")] = None,
    precision: Annotated[
        str | None, typer.Option(help="Force precision, for example fp16 or int8.")
    ] = None,
    hardware_profile: Annotated[
        str | None,
        typer.Option(
            "--hardware-profile",
            help="Use a bundled hardware profile instead of local detection.",
        ),
    ] = None,
    catalog: Annotated[
        Path | None, typer.Option(help="Optional custom model catalog JSON (schema v1 or v2).")
    ] = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Use verified cache or bundled fallback only.")
    ] = False,
    limit: Annotated[
        int, typer.Option(help="Maximum number of candidates to show.", min=1)
    ] = 8,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Rank models against the current device or a known hardware profile."""
    task = task.strip().lower()
    if task not in {"detection", "vlm"}:
        raise typer.BadParameter("task must be detection or vlm", param_hint="--task")
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
    )
    try:
        items = recommend_models(
            hardware,
            constraints,
            catalog_path=catalog,
            offline=offline,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RegistryError) as exc:
        if catalog is None:
            raise typer.BadParameter(f"registry error: {exc}") from exc
        raise typer.BadParameter(f"invalid catalog: {exc}", param_hint="--catalog") from exc
    if json_output:
        print_recommendations(items, limit=limit, as_json=True)
    else:
        print_hardware(hardware)
        console.print()
        print_recommendations(items, limit=limit)


@app.command("catalog")
def catalog_command(
    task: Annotated[str | None, typer.Option(help="Filter to detection or vlm.")] = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Use verified cache or bundled fallback only.")
    ] = False,
) -> None:
    """List model profiles from the official registry or offline fallback."""
    if task:
        task = task.strip().lower()
        if task not in {"detection", "vlm"}:
            raise typer.BadParameter("task must be detection or vlm", param_hint="--task")
    loaded = load_model_catalog(offline=offline)
    models = list(loaded.models)
    if task:
        models = [model for model in models if model.task == task]
    table = Table(title="Model registry")
    table.add_column("ID")
    table.add_column("Task")
    table.add_column("Parameters")
    table.add_column("Accuracy")
    table.add_column("Runtimes")
    table.add_column("Verification")
    for model in models:
        accuracy = "-"
        if model.accuracy:
            accuracy = f"{model.accuracy.value:g} {model.accuracy.name}"
        table.add_row(
            model.id,
            model.task,
            f"{model.params_m:g}M",
            accuracy,
            ", ".join(model.runtimes),
            model.verification_status,
        )
    console.print(table)
    console.print(
        f"Registry v{loaded.provenance.registry_version or '?'} · {loaded.provenance.source}"
    )
    if loaded.provenance.warning:
        console.print(f"Warning: {loaded.provenance.warning}")


@app.command("profiles")
def profiles_command() -> None:
    """List bundled hardware profiles with published benchmark coverage."""
    profiles = load_hardware_profiles()
    table = Table(title="Hardware profiles")
    table.add_column("ID")
    table.add_column("Hardware")
    table.add_column("Memory")
    table.add_column("Platform")
    for profile_id, item in sorted(profiles.items()):
        table.add_row(profile_id, item["display_name"], f"{item['memory_gb']} GB", item["platform"])
    console.print(table)


@registry_app.command("status")
def registry_status(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Show local registry cache and trust state without network access."""
    _print_registry_status(RegistryClient().status(), json_output)


@registry_app.command("update")
def registry_update(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Fetch and verify the latest official registry now."""
    try:
        snapshot = RegistryClient().update()
    except RegistryError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = snapshot.provenance.to_dict()
    payload["model_count"] = len(snapshot.models)
    if json_output:
        console.print_json(json.dumps(payload))
    else:
        console.print(
            f"Updated to registry v{snapshot.provenance.registry_version} "
            f"({len(snapshot.models)} models), Sigstore verified."
        )


@registry_app.command("clear-cache")
def registry_clear_cache(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Remove cached registry data while preserving rollback-protection state."""
    client = RegistryClient()
    removed = client.clear_cache()
    payload = {"removed": removed, "security_state_preserved": True}
    if json_output:
        console.print_json(json.dumps(payload))
    else:
        console.print(f"Removed {len(removed)} cache file(s). Rollback trust state preserved.")


@app.command()
def benchmark(
    model: Annotated[Path, typer.Argument(help="Path to a local ONNX model.")],
    iterations: Annotated[int, typer.Option(help="Timed inference iterations.")] = 50,
    warmup: Annotated[int, typer.Option(help="Warm-up iterations.")] = 10,
    shape: Annotated[
        str | None, typer.Option(help="Override a dynamic input shape, for example 1,3,640,640.")
    ] = None,
    provider: Annotated[
        str | None, typer.Option(help="ONNX Runtime execution provider override.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the benchmark result to JSON.")
    ] = None,
) -> None:
    """Benchmark a local ONNX model on the actual machine."""
    if not model.exists():
        raise typer.BadParameter(f"model does not exist: {model}")
    if iterations < 1 or warmup < 0:
        raise typer.BadParameter("iterations must be >= 1 and warmup must be >= 0")
    shape_override = _parse_shape(shape)
    hardware = detect_hardware()
    try:
        result = benchmark_onnx(
            model,
            platform_kind=hardware.platform,
            iterations=iterations,
            warmup=warmup,
            shape_override=shape_override,
            provider=provider,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(json.dumps(result.to_dict()))
    if output:
        save_result(result, output)
        console.print(f"Saved {output}")


if __name__ == "__main__":
    app()
