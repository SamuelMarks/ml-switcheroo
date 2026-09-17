.PHONY: all help build package docs docs_all

all: help

help:
	@echo "Available targets:"
	@echo "  build              - Install the package locally (editable)"
	@echo "  package            - Build the sdist and wheel distributions"
	@echo "  docs               - Build only the homepage for fast iteration"
	@echo "  docs_all           - Build the full documentation (incl. operators reference)"

PYTHON ?= $(shell test -f .venv/bin/python && echo .venv/bin/python || echo python3)

build:
	uv pip install -e .

package:
	uv build

docs:
	$(PYTHON) scripts/build_docs.py

docs_all:
	BUILD_ALL_DOCS=1 $(PYTHON) scripts/build_docs.py --build-all
