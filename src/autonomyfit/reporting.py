from __future__ import annotations

import json
from dataclasses import asdict

from rich.console import Console
from rich.table import Table

from .models import HardwareProfile, Recommendation

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
    table.add_row("RAM", f"{hardware.ram_available_gb:.1f} GB available / {hardware.ram_total_gb:.1f} GB total")
    table.add_row("Accelerator", hardware.gpu or "None detected")
    if hardware.accelerator_memory_gb is not None:
        label = "unified available" if hardware.unified_memory else "VRAM"
        table.add_row("Accelerator memory", f"{hardware.accelerator_memory_gb:.1f} GB ({label})")
    if hardware.jetpack:
        table.add_row("JetPack / L4T", hardware.jetpack)
    if hardware.power_mode:
        table.add_row("Power mode", hardware.power_mode)
    if hardware.matched_profile:
        table.add_row("Benchmark profile", hardware.matched_profile)
    console.print(table)

    runtime_table = Table(title="Runtime readiness")
    runtime_table.add_column("Runtime")
    runtime_table.add_column("Available / supported")
    runtime_table.add_column("Version")
    runtime_table.add_column("Backend")
    for capability in hardware.runtimes:
        runtime_table.add_row(
            capability.name,
            "yes" if capability.available else "no",
            capability.version or "-",
            capability.detail or "-",
        )
    console.print(runtime_table)


def _recommendation_dict(item: Recommendation) -> dict[str, object]:
    data = {
        "model_id": item.model.id,
        "display_name": item.model.display_name,
        "task": item.model.task,
        "verdict": item.verdict,
        "score": item.score,
        "runtime": item.runtime,
        "precision": item.precision,
        "runtime_available": item.runtime_available,
        "memory_gb": round(item.estimated_memory_gb, 3),
        "memory_evidence": item.memory_evidence,
        "latency_ms": round(item.latency_ms, 3) if item.latency_ms is not None else None,
        "fps": round(item.fps, 2) if item.fps is not None else None,
        "reasons": list(item.reasons),
        "blockers": list(item.blockers),
    }
    if item.model.accuracy:
        data["accuracy"] = asdict(item.model.accuracy)
    if item.benchmark:
        data["benchmark_source"] = item.benchmark.source_url
    return data


def print_recommendations(items: list[Recommendation], limit: int = 8, as_json: bool = False) -> None:
    selected = items[:limit]
    if as_json:
        console.print_json(json.dumps([_recommendation_dict(item) for item in selected]))
        return

    table = Table(title="Model fit")
    table.add_column("Model", style="bold")
    table.add_column("Verdict")
    table.add_column("Runtime")
    table.add_column("Memory")
    table.add_column("Latency")
    table.add_column("FPS")
    table.add_column("Accuracy")

    for item in selected:
        accuracy = "-"
        if item.model.accuracy:
            accuracy = f"{item.model.accuracy.value:.1f} {item.model.accuracy.name}"
        latency = f"{item.latency_ms:.2f} ms" if item.latency_ms is not None else "unmeasured"
        fps = f"{item.fps:.1f}" if item.fps is not None else "-"
        memory_suffix = "pub" if item.memory_evidence == "published" else "est"
        table.add_row(
            item.model.display_name,
            item.verdict,
            f"{item.runtime}/{item.precision}",
            f"{item.estimated_memory_gb:.2f} GB {memory_suffix}",
            latency,
            fps,
            accuracy,
        )
    console.print(table)

    if selected:
        top = selected[0]
        console.print(f"\n[bold]Top candidate[/bold]  {top.model.display_name}")
        for reason in top.reasons:
            console.print(f"  + {reason}")
        for blocker in top.blockers:
            console.print(f"  - {blocker}")
        if top.verdict == "BENCHMARK_REQUIRED":
            console.print(
                "  Benchmark on this exact device before accepting latency, FPS, or power constraints."
            )
