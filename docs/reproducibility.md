# Reproducibility

A performance number is meaningful only with its model, artifact, system and measurement context.

For local evidence, preserve the generated AutonomyFit benchmark report. It records the model ID/revision, SHA-256 of the exact artifact, hardware fingerprint/profile, OS/architecture/driver fields, runtime and provider versions, precision/quantization labels, batch/input shape, warmup and iteration settings, deterministic random seed, load time, latency distribution, throughput, memory scope, power scope, thermal readings where exposed and an environment fingerprint.

Use the same exact artifact when comparing results. A different artifact hash or known model revision is treated as different evidence. A runtime-version mismatch is not silently reused as exact evidence.

Power modes, thermal state, clocks and background workload can materially change results. AutonomyFit records what it can observe but does not claim that missing telemetry is constant.

Vendor tables and standardized suites should retain their original scenario definitions. In particular, do not translate a benchmark from a different model/workload into an AutonomyFit `VERIFIED_FIT` claim merely because the model family or task is similar.
