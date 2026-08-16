.PHONY: install test lint smoke registry-validate discovery-test

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

registry-validate:
	python scripts/validate_registry.py registry/source/registry-v2.json

discovery-test:
	pytest -q tests/test_discovery.py

smoke:
	python -m autonomyfit scan --json >/dev/null
	python -m autonomyfit registry status --json >/dev/null
	python -m autonomyfit models --offline --json >/dev/null
	python -m autonomyfit search smolvlm --offline --json >/dev/null
	python -m autonomyfit info smolvlm-256m-instruct --offline --json >/dev/null
	python -m autonomyfit recommend --offline --hardware-profile jetson-orin-nx-16gb --fps 200 --latency-ms 5 --json >/dev/null
