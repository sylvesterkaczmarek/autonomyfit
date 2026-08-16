from __future__ import annotations

import json
from dataclasses import asdict

from rich.console import Console
from rich.table import Table

from .evidence import evidence_to_dict
from .models import HardwareProfile, Recommendation, RegistryProvenance

console = Console()


def print_hardware(hardware: HardwareProfile, as_json: bool = False) -> None:
    if as_json:
        console.print_json(json.dumps(hardware.to_dict()))
        return
    table = Table(title="Detected hardware", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Platform", hardware.platform)
    table.add_row("OS", f"{hardware.os_name} / {hardware.architecture}")
    table.add_row("CPU", hardware.cpu)
    table.add_row("Accelerator type", hardware.accelerator_type)
    table.add_row("Memory topology", hardware.memory_topology)
    table.add_row(
        "RAM",
        f"{hardware.ram_available_gb:.1f} GB available / {hardware.ram_total_gb:.1f} GB total",
    )
    table.add_row("Accelerator", hardware.gpu or "None detected")
    if hardware.accelerator_memory_gb is not None:
        label = "unified available" if hardware.unified_memory else "VRAM"
        table.add_row("Accelerator memory", f"{hardware.accelerator_memory_gb:.1f} GB ({label})")
    if hardware.supported_precisions:
        table.add_row("Precision capability", ", ".join(hardware.supported_precisions))
    if hardware.jetpack:
        table.add_row("JetPack / L4T", hardware.jetpack)
    if hardware.driver:
        table.add_row("Driver", hardware.driver)
    if hardware.power_mode:
        table.add_row("Power mode", hardware.power_mode)
    if hardware.matched_profile:
        table.add_row("Hardware profile", hardware.matched_profile)
    console.print(table)

    runtime_table = Table(title="Runtime readiness")
    runtime_table.add_column("Runtime")
    runtime_table.add_column("Available")
    runtime_table.add_column("Version")
    runtime_table.add_column("Evidence")
    runtime_table.add_column("Backend / provider")
    for capability in hardware.runtimes:
        runtime_table.add_row(
            capability.name,
            "yes" if capability.available else "no",
            capability.version or "-",
            "verified capability" if capability.verified else "coverage unverified",
            capability.provider or capability.detail or "-",
        )
    console.print(runtime_table)


def _registry_dict(value: RegistryProvenance | None) -> dict[str, object] | None:
    return value.to_dict() if value else None


def recommendation_dict(item: Recommendation) -> dict[str, object]:
    data: dict[str, object] = {
        "model_id": item.model.id,
        "display_name": item.model.display_name,
        "family": item.model.family,
        "task": item.model.task,
        "verdict": item.verdict,
        "score": item.score,
        "objective": item.objective,
        "objective_rank": item.objective_rank,
        "pareto_rank": item.pareto_rank,
        "pareto_frontier": item.pareto_rank == 0,
        "dominates": list(item.dominates),
        "dominated_by": list(item.dominated_by),
        "runtime": item.runtime,
        "precision": item.precision,
        "runtime_available": item.runtime_available,
        "memory_gb": round(item.estimated_memory_gb, 3) if item.estimated_memory_gb is not None else None,
        "memory_evidence": item.memory_evidence,
        "parameters_m": item.model.params_m,
        "latency_ms": round(item.latency_ms, 3) if item.latency_ms is not None else None,
        "fps": round(item.fps, 2) if item.fps is not None else None,
        "confidence": item.confidence.to_dict() if item.confidence else None,
        "evidence_confidence": item.evidence_confidence,
        "hard_constraints_passed": not bool(item.blockers),
        "reasons": list(item.reasons),
        "blockers": list(item.blockers),
        "measured": list(item.measured),
        "estimated": list(item.estimated),
        "unknowns": list(item.unknowns),
        "next_benchmark": item.next_benchmark,
        "registry": _registry_dict(item.registry),
        "model_provenance": {
            "source_id": item.model.source_id,
            "source_url": item.model.source_url,
            "source_revision": item.model.source_revision,
            "release_date": item.model.release_date,
            "last_checked": item.model.last_checked,
            "last_verified": item.model.last_verified,
            "verification_status": item.model.verification_status,
            "license_spdx": item.model.license_spdx,
            "license_status": item.model.license_status,
            "experimental": (
                item.model.experimental or item.model.verification_status == "discovered"
            ),
        },
    }
    if item.model.accuracy:
        data["accuracy"] = asdict(item.model.accuracy)
    if item.benchmark:
        data["benchmark_evidence"] = evidence_to_dict(item.benchmark)
        data["benchmark_match"] = {
            "exact": bool(item.evidence_match and item.evidence_match.exact),
            "identity_complete": bool(item.evidence_match and item.evidence_match.identity_complete),
            "limitations": list(item.evidence_match.reasons) if item.evidence_match else [],
        }
    return data


def print_recommendations(items: list[Recommendation], limit: int = 8, as_json: bool = False) -> None:
    selected = items[:limit]
    if as_json:
        console.print_json(json.dumps([recommendation_dict(item) for item in selected]))
        return

    table = Table(title="Model fit")
    table.add_column("#")
    table.add_column("Model", style="bold")
    table.add_column("Verdict")
    table.add_column("Pareto")
    table.add_column("Runtime")
    table.add_column("Memory")
    table.add_column("Latency evidence")
    table.add_column("Confidence")
    for item in selected:
        if item.latency_ms is None:
            latency = "unknown"
        elif item.evidence_match and item.evidence_match.exact:
            latency = f"{item.latency_ms:.2f} ms exact"
        else:
            latency = f"{item.latency_ms:.2f} ms reference"
        if item.estimated_memory_gb is None:
            memory = "unknown"
        else:
            suffix = "pub" if item.memory_evidence == "published" else "est"
            memory = f"{item.estimated_memory_gb:.2f} GB {suffix}"
        confidence = (
            f"{item.confidence.score:.0f} {item.confidence.level}" if item.confidence else "unknown"
        )
        table.add_row(
            str(item.objective_rank or "-"),
            item.model.display_name,
            item.verdict,
            str(item.pareto_rank if item.pareto_rank is not None else "-"),
            f"{item.runtime}/{item.precision}",
            memory,
            latency,
            confidence,
        )
    console.print(table)

    if not selected:
        return
    top = selected[0]
    if top.registry:
        trust = "verified" if top.registry.signature_verified else "package/custom trust"
        stale = " · stale" if top.registry.stale else ""
        version = f"v{top.registry.registry_version}" if top.registry.registry_version else "v?"
        console.print(f"\n[bold]Registry[/bold]  {version} · {top.registry.source} · {trust}{stale}")
    console.print(
        f"\n[bold]Top candidate[/bold]  {top.model.display_name} · objective={top.objective} · "
        f"Pareto layer {top.pareto_rank if top.pareto_rank is not None else '-'}"
    )
    if top.confidence:
        console.print(
            f"  Confidence: {top.confidence.score:.1f}/100 ({top.confidence.level}); "
            f"quantity coverage {top.confidence.quantity_coverage:.2f}"
        )
    for reason in top.reasons:
        console.print(f"  + {reason}")
    for blocker in top.blockers:
        console.print(f"  - {blocker}")
    for unknown in top.unknowns:
        console.print(f"  ? {unknown}")
    if top.dominates:
        console.print("  Dominates: " + ", ".join(top.dominates))
    if top.dominated_by:
        console.print("  Dominated by: " + ", ".join(top.dominated_by))
    if top.next_benchmark:
        console.print("  Reduce uncertainty: " + top.next_benchmark)
    console.print(
        "  Model source: "
        + top.model.source_url
        + (f" @ {top.model.source_revision}" if top.model.source_revision else " @ revision unknown")
    )
