.PHONY: all help build package docs docs_only_homepage

all: help

help:
	@echo "Available targets:"
	@echo "  build              - Install the package locally (editable)"
	@echo "  package            - Build the sdist and wheel distributions"
	@echo "  docs               - Build the documentation"
	@echo "  docs_only_homepage - Build only the homepage for fast iteration"

build:
	uv pip install -e .

package:
	uv build

docs:
	python3 scripts/build_docs.py

docs_only_homepage:
	python3 scripts/build_docs.py --homepage-only
