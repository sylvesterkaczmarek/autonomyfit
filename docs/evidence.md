# Evidence model

AutonomyFit 0.4 separates model metadata from deployment evidence. A model name is not a benchmark identity.

The normalized evidence store contains explicit entities for model revisions, artifacts, accuracy, compatibility, memory, benchmark results, hardware profiles, runtime profiles and software stacks. Benchmark evidence records the model revision and artifact SHA-256 when known, the hardware/runtime/precision tuple, software context, input shape, measurement procedure, latency statistics, throughput, memory scope, power scope and source date.

## Evidence quality

AutonomyFit uses five evidence classes:

1. `local-measured` - produced by AutonomyFit on the exact local artifact.
2. `standardized` - a standardized benchmark such as an applicable VALID MLPerf result with exact model/artifact identity supplied.
3. `vendor-published` - a vendor table or benchmark report.
4. `third-party-reproducible` - independently reproducible external evidence.
5. `metadata-estimate` - an estimate derived from metadata rather than measurement.

Only identity-complete `local-measured` or `standardized` evidence can establish `VERIFIED_FIT`. Vendor evidence remains useful for screening and context, but does not silently become proof for a different artifact.

## Exact matching

Performance evidence is matched on model ID, hardware identity, runtime and precision. Known revision, artifact-hash or runtime-version mismatches reject the evidence. Missing identity fields downgrade a result to context-only evidence. Stale evidence is also prevented from becoming exact proof.

This is intentionally strict. A friendly model name such as `YOLO26n` is not enough to prove that two binaries, engine builds or weight revisions are equivalent.

## Current vendor evidence

The bundled Jetson YOLO26n table is normalized as `vendor-published`. It contains useful per-device/runtime latency values, but the upstream table does not identify one exact model revision and artifact SHA-256 for those rows. AutonomyFit therefore shows values such as 4.13 ms on Jetson Orin NX as reference evidence and returns `BENCHMARK_REQUIRED` when a latency/FPS constraint needs proof.

## Power and memory scope

Power measurements carry an explicit scope. Jetson VDD_IN rail readings and NVIDIA GPU `power.draw` are not treated as whole-system wall power. MLPerf power measurements use their own standardized system-level methodology.

The generic local memory sampler reports process RSS. It is not labelled as accelerator-memory consumption. Native backend-specific accelerator memory can be added later without conflating the two quantities.

## MLPerf

AutonomyFit includes strict MLPerf summary parsing/import support, but does not ship a standardized MLPerf result unless the MLPerf workload, exact model revision and artifact identity are genuinely applicable. No MLPerf result is currently mapped onto YOLO26 merely because another object detector appears in MLPerf.
