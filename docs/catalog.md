# Model catalog

A custom catalog is JSON with a top-level `models` array.

Minimal example:

```json
{
  "schema_version": 1,
  "models": [
    {
      "id": "my-detector",
      "display_name": "My Detector",
      "family": "Internal",
      "task": "detection",
      "params_m": 8.2,
      "runtimes": ["onnx", "tensorrt"],
      "source_id": "internal-benchmark-2026-08",
      "source_url": "https://example.com/evidence"
    }
  ]
}
```

## Required fields

- `id` unique machine-readable identifier
- `display_name` human-readable model name
- `family` model family
- `task` either `detection` or `vlm`
- `params_m` parameter count in millions
- `runtimes` one or more supported runtime identifiers
- `source_id` stable evidence identifier
- `source_url` source for the model metadata

## Optional fields

- `flops_b` forward-pass FLOPs in billions
- `input_size` nominal square image input size
- `accuracy` object containing `name`, `value`, and optional `dataset`
- `published_memory_gb` upstream memory requirement
- `memory_scope` conditions attached to that memory value
- `notes` concise qualification

Custom catalogs replace the bundled model list for that invocation. Bundled hardware benchmark records remain available, but they match only when `id` values correspond exactly.
