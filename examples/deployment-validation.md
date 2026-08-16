# Example: measured detection deployment

This example starts from recommendation and ends with exact local evidence and a reproducible report.

```bash
pip install 'autonomyfit[deployment,benchmark]'

autonomyfit scan

autonomyfit recommend \
  --task detection \
  --objective latency \
  --max-memory-gb 8 \
  --top 3
```

Export or obtain a self-contained ONNX artifact from a source you are permitted to use. Pin its upstream revision and, when available, expected digest.

```bash
autonomyfit validate yolo26n \
  --artifact ./yolo26n.onnx \
  --revision UPSTREAM_COMMIT \
  --sha256 EXPECTED_SHA256 \
  --runtime onnx \
  --benchmark \
  --latency-ms 10 \
  --fps 100 \
  --report ./reports/yolo26n.json \
  --markdown ./reports/yolo26n.md
```

Inspect the local evidence and rerun model selection:

```bash
autonomyfit local-results

autonomyfit recommend \
  --task detection \
  --runtime onnx \
  --latency-ms 10 \
  --fps 100 \
  --objective latency
```

The exact local benchmark can now outrank generic reference evidence for the same identity. If the driver/runtime major version changes or the result becomes stale, it is excluded rather than silently reused.

Render the report again at any time:

```bash
autonomyfit report ./reports/yolo26n.json
```
