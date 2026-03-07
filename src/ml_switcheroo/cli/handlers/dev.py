"""
CLI handlers for dev module.

Provides handlers for development utilities: Matrix rendering, Doc generation,
and Test generation.
"""

from pathlib import Path

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.utils.console import log_info, log_success


def handle_matrix() -> int:
  """
  Handles 'matrix' command.
  Prints the compatibility table to stdout.

  Returns:
      int: 0 on success.
  """
  semantics = SemanticsManager()  # pragma: no cover
  matrix = CompatibilityMatrix(semantics)  # pragma: no cover
  matrix.render()  # pragma: no cover
  return 0  # pragma: no cover


def handle_docs(source: str, target: str, out_path: Path) -> int:
  """
  Handles 'gen-docs' command.
  Generates a migration guide Markdown file.

  Args:
      source: Source framework key.
      target: Target framework key.
      out_path: Output file path.

  Returns:
      int: 0 on success.
  """
  semantics = SemanticsManager()  # pragma: no cover
  log_info(f"Generating comparison: {source} -> {target} at {out_path}...")  # pragma: no cover
  generator = MigrationGuideGenerator(semantics)  # pragma: no cover
  markdown = generator.generate(source, target)  # pragma: no cover
  with open(out_path, "w", encoding="utf-8") as f:  # pragma: no cover
    f.write(markdown)  # pragma: no cover
  log_success(f"Documentation saved to [path]{out_path}[/path]")  # pragma: no cover
  return 0  # pragma: no cover


def handle_gen_tests(out: Path) -> int:
  """
  Handles 'gen-tests' command.
  Generates physical test files for known APIs.

  Args:
      out: Output file path.

  Returns:
      int: 0 on success.
  """
  mgr = SemanticsManager()  # pragma: no cover
  semantics = mgr.get_known_apis()  # pragma: no cover
  out.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
  gen = TestCaseGenerator(semantics_mgr=mgr)  # pragma: no cover
  gen.generate(semantics, out)  # pragma: no cover
  return 0  # pragma: no cover
