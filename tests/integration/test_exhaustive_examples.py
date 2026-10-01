"""Test suite verifying transpilation of all examples into all target frameworks."""

from pathlib import Path
from typing import List, Tuple

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

EXAMPLES_DIR: Path = Path(__file__).parent.parent / "examples"


def _get_example_files() -> List[Tuple[str, str]]:
  """Discovers all example files and extracts their source framework.

  Returns:
    List[Tuple[str, str]]: A list of tuples containing the filename and the source framework name.
  """
  result: List[Tuple[str, str]] = []
  if not EXAMPLES_DIR.exists():
    return result

  for path in EXAMPLES_DIR.glob("ex*.py"):
    basename = path.name
    parts = basename.split(".")
    if len(parts) >= 3:
      source = parts[-2]
      result.append((basename, source))
  return result


EXAMPLE_FILES: List[Tuple[str, str]] = _get_example_files()

TARGETS: List[str] = [
  "torch",
  "jax",
  "mlx",
  "keras",
  "flax_nnx",
  "tensorflow",
  "numpy",
  "paxml",
  "stablehlo",
  "mlir",
  "nvidia_sass",
  "rdna",
  "html",
  "latex_dsl",
  "tikz",
]


@pytest.fixture(scope="module")
def semantics() -> SemanticsManager:
  """Provides a SemanticsManager instance for the tests.

  Returns:
    SemanticsManager: A pre-configured semantics manager.
  """
  return SemanticsManager()


@pytest.mark.parametrize("filename, source", EXAMPLE_FILES)
@pytest.mark.parametrize("target", TARGETS)
def test_exhaustive_example_translation(semantics: SemanticsManager, filename: str, source: str, target: str) -> None:
  """Verifies the transpilation of an example file into a target framework.

  Args:
    semantics (SemanticsManager): The shared semantics manager.
    filename (str): The name of the example file.
    source (str): The source framework extracted from the filename.
    target (str): The target framework to transpile into.
  """
  path: Path = EXAMPLES_DIR / filename
  code: str = path.read_text(encoding="utf-8")
  config = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(code)
  assert result.success, f"Failed converting {filename} to {target}: {result.errors}"
