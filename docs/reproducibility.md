# Reproducibility

AutonomyFit keeps data provenance and local measurements distinct.

For reproducible recommendations record:

- AutonomyFit package version
- registry version and registry source
- model upstream revision when available
- hardware profile/detection output
- runtime and precision
- constraints
- benchmark source or local result

Use JSON output for machine-readable capture:

```bash
autonomyfit scan --json
autonomyfit registry status --json
autonomyfit recommend --offline --hardware-profile jetson-orin-nx-16gb --json
```

`--offline` is useful when reproducing a run against an already verified cache without allowing
a network refresh during the experiment.

The local ONNX benchmark uses deterministic synthetic inputs for execution-cost measurement.
It does not validate task accuracy.
