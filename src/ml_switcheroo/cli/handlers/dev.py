"""CLI handlers for dev module."""

from pathlib import Path

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.utils.console import log_info, log_success


def handle_matrix() -> int:
  """Handles 'matrix' command."""
  semantics = SemanticsManager()
  matrix = CompatibilityMatrix(semantics)
  matrix.render()
  return 0


def handle_docs(source: str, target: str, out_path: Path) -> int:
  """Handles 'gen-docs' command."""
  semantics = SemanticsManager()
  log_info(f"Generating comparison: {source} -> {target} at {out_path}...")
  generator = MigrationGuideGenerator(semantics)
  markdown = generator.generate(source, target)
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(markdown)
  log_success(f"Documentation saved to [path]{out_path}[/path]")
  return 0


def handle_gen_tests(out: Path) -> int:
  """Handles 'gen-tests' command."""
  mgr = SemanticsManager()
  semantics = mgr.get_known_apis()
  out.parent.mkdir(parents=True, exist_ok=True)
  gen = TestGenerator()
  gen.generate(semantics, out)
  return 0
