# Evidence rules

AutonomyFit separates registry metadata from hardware-specific benchmark evidence.

## Model metadata

Registry Schema v2 records upstream provenance, licence, compatibility declarations, accuracy
and memory evidence independently. Every published value retains a source URL and source ID.
Unknown quantities remain unknown.

Current bundled exact object-detection benchmark evidence is limited to YOLO26n Jetson tuples
published by Ultralytics. A result is used only when hardware profile, model ID, runtime and
precision all match.

SmolVLM memory entries are sourced from their upstream Hugging Face model cards and are used
only as memory screens. Their throughput remains unmeasured unless a hardware-specific record
exists.

## Estimates

When no published inference-memory figure exists, AutonomyFit computes a conservative screen
from parameter count, precision and workload class. This is labelled `screening estimate` and
is not a peak-memory measurement.

## Registry provenance

The registry itself carries generation, expiry and verification timestamps. Recommendation
JSON includes both registry provenance and per-model upstream provenance so consumers can
distinguish current signed registry data, cache, bundled fallback and custom inputs.
