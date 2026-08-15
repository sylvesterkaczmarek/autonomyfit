# Reproducibility

## Deterministic behavior

Model ranking is deterministic for a fixed hardware profile, catalog and constraint set. The bundled catalogs are static package data.

The ONNX benchmark creates input tensors with NumPy random generator seed `0`. Latency itself is not deterministic because scheduling, clock state, thermal state and background load vary.

## Repeatable comparison

For a useful device comparison:

1. Use the same model file and input shape.
2. Use the same runtime provider and precision.
3. Record the device power mode and clock configuration.
4. Close unrelated compute-heavy processes.
5. Use the same warm-up and iteration counts.
6. Repeat the benchmark after the device reaches a stable thermal state.
7. Preserve the JSON output with the model artefact hash in your experiment record.

AutonomyFit does not change clocks or power modes automatically.

## Bundled evidence

Bundled latency values are copied only from upstream tables that expose the relevant model/hardware/runtime conditions. They are not re-labelled as local measurements.

## Clean checkout check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python -m autonomyfit scan --json
python -m autonomyfit recommend --hardware-profile jetson-orin-nx-16gb --fps 200 --latency-ms 5 --json
```
