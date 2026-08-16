.PHONY: install test lint smoke registry-validate

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

registry-validate:
	python scripts/validate_registry.py registry/source/registry-v2.json

smoke:
	python -m autonomyfit scan --json >/dev/null
	python -m autonomyfit registry status --json >/dev/null
	python -m autonomyfit recommend --offline --hardware-profile jetson-orin-nx-16gb --fps 200 --latency-ms 5 --json >/dev/null
