# Registry Schema v2

The official model registry is a separately versioned data product. The Python engine supports
schema version 2 and ships a small fallback snapshot of that schema.

Each model separates:

- stable model ID, display name, family and variant
- task and input/output modalities
- upstream source, revision, release date and last-checked timestamp
- parameter count, FLOPs and input information
- runtimes, precisions and quantizations
- upstream licence metadata
- accuracy, memory, compatibility and benchmark evidence references
- verification state and last-verified timestamp

The machine-readable JSON Schema is packaged at
`src/autonomyfit/data/registry-v2.schema.json`.

## Versioning

`schema_version` changes only when the document shape becomes incompatible.
`registry.registry_version` increases for every official data change. A model-data change does
not require a PyPI package version change.

## Legacy custom catalogues

Schema-v1 custom catalogues remain supported for backward compatibility. Example:

```json
{
  "schema_version": 1,
  "models": [
    {
      "id": "my-detector",
      "display_name": "My Detector",
      "family": "custom",
      "task": "detection",
      "params_m": 8.0,
      "runtimes": ["onnx", "tensorrt"],
      "source_id": "internal-test",
      "source_url": "https://example.com/model"
    }
  ]
}
```

Use it with:

```bash
autonomyfit recommend --catalog examples/custom-models.json --task detection
```

Local custom catalogues are not presented as official signed registry data.
