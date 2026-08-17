from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .artifacts import ArtifactError, ArtifactManager
from .catalog import load_model_catalog
from .deployment import (
    DeploymentValidationError,
    ValidationOptions,
    assess_candidates,
    validate_deployment,
)
from .deployment_reports import (
    DeploymentReportError,
    load_deployment_report,
    render_deployment_markdown,
    save_deployment_report,
)
from .evidence import local_benchmark_dir
from .hardware import detect_hardware
from .local_results import LOCAL_RESULT_MAX_AGE_DAYS, list_local_results
from .registry import RegistryError


def _shape(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        dims = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("shape must be comma-separated positive integers") from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise typer.BadParameter("shape dimensions must be positive integers")
    return dims


def _find_model(model_id: str, offline: bool):
    loaded = load_model_catalog(offline=offline)
    needle = model_id.casefold()
    model = next(
        (
            item
            for item in loaded.models
            if item.id.casefold() == needle or item.display_name.casefold() == needle
        ),
        None,
    )
    if model is None:
        raise typer.BadParameter(f"unknown model: {model_id}")
    return model


def register_deployment_commands(app: typer.Typer, console: Console) -> None:
    @app.command("artifacts")
    def artifacts_command(
        model_id: Annotated[str, typer.Argument(help="Canonical model ID.")],
        revision: Annotated[str | None, typer.Option(help="Revision/tag/commit to inspect.")] = None,
        offline: Annotated[bool, typer.Option("--offline", help="Inspect verified local cache only.")] = False,
        json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
    ) -> None:
        """Discover safe static artifacts and exact upstream revision identity without executing repository code."""
        try:
            model = _find_model(model_id, offline)
            manager = ArtifactManager()
            payload = manager.discover_huggingface(model, revision=revision, offline=offline)
        except (ArtifactError, RegistryError, OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        payload["model_id"] = model.id
        payload["license_spdx"] = model.license_spdx
        payload["license_status"] = model.license_status
        if json_output:
            console.print_json(json.dumps(payload))
            return
        table = Table(title=f"Artifacts for {model.id}")
        table.add_column("File")
        table.add_column("Safe automatic handling")
        table.add_column("Size")
        table.add_column("Reason")
        for item in payload.get("candidates", []):
            size = item.get("size_bytes")
            table.add_row(
                str(item.get("filename")),
                "yes" if item.get("safe_static") else "no",
                str(size) if size is not None else "unknown",
                str(item.get("reason")),
            )
        console.print(table)
        console.print(f"Resolved revision: {payload.get('resolved_revision') or 'cache-only'}")
        if payload.get("remote_code_required"):
            console.print("Warning: upstream repository contains custom/remote code; AutonomyFit will not execute it.")

    @app.command("validate")
    def validate_command(
        model_id: Annotated[str, typer.Argument(help="Canonical model ID.")],
        artifact: Annotated[Path | None, typer.Option("--artifact", help="Local artifact path.")] = None,
        fetch: Annotated[bool, typer.Option("--fetch", help="Safely fetch a static artifact from a supported upstream source.")] = False,
        filename: Annotated[str | None, typer.Option(help="Exact upstream artifact filename.")] = None,
        revision: Annotated[str | None, typer.Option(help="Pin revision/tag; resolved to an immutable commit for supported hubs.")] = None,
        sha256: Annotated[str | None, typer.Option("--sha256", help="Expected artifact SHA-256.")] = None,
        artifact_url: Annotated[str | None, typer.Option("--artifact-url", help="HTTPS artifact URL; requires --sha256.")] = None,
        runtime: Annotated[str | None, typer.Option(help="Target runtime/provider path.")] = None,
        precision: Annotated[str | None, typer.Option(help="Target precision.")] = None,
        provider: Annotated[str | None, typer.Option(help="ONNX Runtime provider override.")] = None,
        device: Annotated[str | None, typer.Option(help="Native runtime device, e.g. CPU/GPU/NPU.")] = None,
        compute_units: Annotated[
            str | None,
            typer.Option("--compute-units", help="Core ML compute units: ALL, CPU_ONLY, CPU_AND_GPU or CPU_AND_NE."),
        ] = None,
        convert: Annotated[bool, typer.Option("--convert", help="Persist a supported local conversion before benchmarking.")] = False,
        benchmark: Annotated[bool, typer.Option("--benchmark", help="Run the viable artifact on this machine and import exact local evidence.")] = False,
        iterations: Annotated[int, typer.Option(help="Timed benchmark iterations.", min=1)] = 50,
        warmup: Annotated[int, typer.Option(help="Warm-up count/setting.", min=0)] = 10,
        shape: Annotated[str | None, typer.Option(help="Explicit input shape for supported single-input paths.")] = None,
        allow_restricted_license: Annotated[
            bool,
            typer.Option(
                "--allow-restricted-license",
                help="Acknowledge non-standard/restricted upstream terms before automatic acquisition; this does not grant usage rights.",
            ),
        ] = False,
        trust_artifact: Annotated[
            bool,
            typer.Option("--trust-artifact", help="Explicitly trust a local executable/pickle-style artifact you control."),
        ] = False,
        import_local: Annotated[
            bool,
            typer.Option("--import-local/--no-import-local", help="Import successful benchmark evidence into the local-results layer."),
        ] = True,
        hardware_profile: Annotated[str | None, typer.Option("--hardware-profile")] = None,
        latency_ms: Annotated[float | None, typer.Option("--latency-ms", min=0.0)] = None,
        fps: Annotated[float | None, typer.Option("--fps", min=0.0)] = None,
        power_w: Annotated[float | None, typer.Option("--power-w", min=0.0)] = None,
        max_memory_gb: Annotated[float | None, typer.Option("--max-memory-gb", min=0.0)] = None,
        report: Annotated[Path | None, typer.Option("--report", help="Write the full deployment report as JSON or Markdown by extension.")] = None,
        markdown: Annotated[Path | None, typer.Option("--markdown", help="Also write a readable Markdown report.")] = None,
        offline: Annotated[bool, typer.Option("--offline", help="No network access; use verified registry/artifact caches.")] = False,
        json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
    ) -> None:
        """Validate artifact identity, runtime compatibility, optional conversion and measured deployment constraints."""
        selected_sources = sum((artifact is not None, artifact_url is not None, fetch))
        if selected_sources > 1:
            raise typer.BadParameter("use only one artifact source: --artifact, --artifact-url or --fetch")
        try:
            payload = validate_deployment(
                ValidationOptions(
                    model_id=model_id,
                    artifact=artifact,
                    artifact_url=artifact_url,
                    filename=filename,
                    revision=revision,
                    expected_sha256=sha256,
                    fetch=fetch,
                    offline=offline,
                    runtime=runtime,
                    precision=precision,
                    provider=provider,
                    device=device,
                    compute_units=compute_units,
                    convert=convert,
                    benchmark=benchmark,
                    iterations=iterations,
                    warmup=warmup,
                    shape=_shape(shape),
                    trust_artifact=trust_artifact,
                    import_local=import_local,
                    hardware_profile=hardware_profile,
                    max_latency_ms=latency_ms,
                    min_fps=fps,
                    max_power_w=power_w,
                    max_memory_gb=max_memory_gb,
                    allow_restricted_license=allow_restricted_license,
                )
            )
            if report:
                save_deployment_report(payload, report)
            if markdown:
                save_deployment_report(payload, markdown)
        except (DeploymentValidationError, DeploymentReportError, RegistryError, OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        if json_output:
            console.print_json(json.dumps(payload))
        else:
            console.print(render_deployment_markdown(payload), markup=False)
            if report:
                console.print(f"Saved {report}")
            if markdown:
                console.print(f"Saved {markdown}")

    @app.command("report")
    def report_command(
        source: Annotated[Path, typer.Argument(help="Deployment report JSON.")],
        output: Annotated[Path | None, typer.Option("--output", "-o", help="Write Markdown/JSON by extension.")] = None,
        json_output: Annotated[bool, typer.Option("--json", help="Emit the validated JSON report.")] = False,
    ) -> None:
        """Validate and render a reproducible deployment report."""
        try:
            payload = load_deployment_report(source)
            if output:
                save_deployment_report(payload, output)
        except DeploymentReportError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if json_output:
            console.print_json(json.dumps(payload))
        else:
            console.print(render_deployment_markdown(payload), markup=False)
            if output:
                console.print(f"Saved {output}")

    @app.command("local-results")
    def local_results_command(
        max_age_days: Annotated[
            int,
            typer.Option("--max-age-days", min=1, help="Freshness window for local evidence."),
        ] = LOCAL_RESULT_MAX_AGE_DAYS,
        json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
    ) -> None:
        """Inspect exact-machine local benchmark evidence and stack-change invalidation."""
        hardware = detect_hardware()
        statuses = list_local_results(
            local_benchmark_dir(), hardware, max_age_days=max_age_days
        )
        if json_output:
            console.print_json(json.dumps([item.to_dict() for item in statuses]))
            return
        table = Table(title="Local deployment evidence")
        table.add_column("Benchmark")
        table.add_column("Model")
        table.add_column("Runtime")
        table.add_column("Valid now")
        table.add_column("Reason")
        for item in statuses:
            table.add_row(
                item.benchmark_id or item.path.name,
                item.model_id or "unknown",
                item.runtime or "unknown",
                "yes" if item.valid else "no",
                "; ".join(item.reasons) or "exact-machine evidence is current",
            )
        console.print(table)

    @app.command("assess")
    def assess_command(
        model_ids: Annotated[list[str], typer.Argument(help="Two or more candidate model IDs.")],
        artifacts: Annotated[
            list[str],
            typer.Option("--artifact", help="Repeat MODEL=PATH for every candidate."),
        ],
        runtime: Annotated[str | None, typer.Option(help="Target runtime.")] = None,
        precision: Annotated[str | None, typer.Option(help="Target precision.")] = None,
        iterations: Annotated[int, typer.Option(min=1)] = 50,
        warmup: Annotated[int, typer.Option(min=0)] = 10,
        hardware_profile: Annotated[str | None, typer.Option("--hardware-profile")] = None,
        offline: Annotated[bool, typer.Option("--offline")] = False,
        json_output: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Benchmark candidate artifacts locally, import the measurements, then reorder the candidates."""
        mapping: dict[str, Path] = {}
        for item in artifacts:
            if "=" not in item:
                raise typer.BadParameter("--artifact must use MODEL=PATH")
            key, raw_path = item.split("=", 1)
            mapping[key.strip()] = Path(raw_path).expanduser()
        try:
            payload = assess_candidates(
                model_ids,
                mapping,
                runtime=runtime,
                precision=precision,
                iterations=iterations,
                warmup=warmup,
                offline=offline,
                hardware_profile=hardware_profile,
            )
        except (DeploymentValidationError, RegistryError, OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        if json_output:
            console.print_json(json.dumps(payload))
            return
        table = Table(title="Candidates after local measurement")
        table.add_column("Rank")
        table.add_column("Model")
        table.add_column("Verdict")
        table.add_column("Latency")
        table.add_column("Confidence")
        for index, item in enumerate(payload["reordered_recommendations"], start=1):
            table.add_row(
                str(index),
                str(item.get("model_id", "unknown")),
                str(item.get("verdict", "unknown")),
                str(item.get("latency_ms") or "unknown"),
                str((item.get("confidence") or {}).get("score", "unknown")),
            )
        console.print(table)
