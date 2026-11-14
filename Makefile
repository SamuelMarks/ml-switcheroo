.PHONY: all help build package wheels docs docs_all test

all: help

help:
	@echo "Available targets:"
	@echo "  build              - Install the package locally (editable)"
	@echo "  package            - Build the sdist and wheel distributions"
	@echo "  wheels             - Bundle on-device zero-* and compiler wheels into docs/_static"
	@echo "  docs               - Build only the homepage for fast iteration"
	@echo "  docs_all           - Build the full documentation (incl. operators reference)"
	@echo "  test               - Run the full test suite with 100% coverage enforcement"

PYTHON ?= $(shell test -f .venv/bin/python && echo .venv/bin/python || echo python3)

build:
	uv pip install -e .

package:
	uv build

wheels:
	uv build --wheel
	$(PYTHON) -c "from pathlib import Path; from ml_switcheroo.sphinx_ext.hooks import bundle_on_device_wheels; bundle_on_device_wheels(Path('.').resolve(), Path('docs/_static').resolve())"

docs:
	$(PYTHON) scripts/build_docs.py

docs_all:
	BUILD_ALL_DOCS=1 $(PYTHON) scripts/build_docs.py --build-all

test:
	uv run --all-extras pytest -n auto --cov=src --cov=scripts --cov=docs --cov-fail-under=100 --cov-branch -W error::DeprecationWarning tests
	uv run python scripts/check_function_coverage.py
