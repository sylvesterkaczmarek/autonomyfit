.PHONY: install test lint smoke

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

smoke:
	python -m autonomyfit scan --json >/dev/null
	python -m autonomyfit recommend --hardware-profile jetson-orin-nx-16gb --fps 200 --latency-ms 5 --json >/dev/null
