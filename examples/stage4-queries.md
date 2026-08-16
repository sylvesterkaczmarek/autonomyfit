# Stage 4 query examples

```bash
# Small segmentation candidates for Apple Silicon
autonomyfit recommend --task segmentation --hardware-profile apple-m4-pro-24gb --objective memory --max-params-m 35 --top 5

# Accuracy-oriented detection screen on a T4
autonomyfit recommend --task detection --hardware-profile nvidia-t4-16gb --objective accuracy --max-params-m 35 --top 8

# Snapdragon path; QNN is selected only where the registry lists QNN support
autonomyfit recommend --task detection --hardware-profile qualcomm-snapdragon-x-elite-16gb --runtime qnn --top 5

# Compare two detector families under the same target
autonomyfit compare yolo26n rfdetr-nano rfdetr-small --hardware-profile nvidia-t4-16gb --objective balanced

# Commercial/licence-aware filtering
autonomyfit recommend --task classification --license Apache-2.0 --objective accuracy --top 5

# Surface preview/experimental entries intentionally
autonomyfit models --task pose --include-experimental
```
