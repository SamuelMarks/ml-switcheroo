"""Test suite for the Cli Visuals module."""

import pathlib
from tests.conftest import SnapshotAssert
from pydantic import JsonValue
from unittest.mock import patch

from rich.console import Console

from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager


class StableMockSemantics(SemanticsManager):
  """Test helper class StableMockSemantics."""

  def get_known_apis(self) -> dict[str, dict[str, JsonValue]]:
    """Mock implementation of get known apis.

    Returns:
        Computed test result or fixture value.
    """
    return {
      "abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}},
      "magic_op": {
        "std_args": ["x"],
        "variants": {"torch": {"api": "torch.magic"}, "jax": {"requires_plugin": "magic_fix"}},
      },
      "unsupported_op": {"std_args": ["x"], "variants": {"torch": {"api": "torch.oops"}}},
    }

  def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        api_name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if api_name == "torch.abs":
      return ("abs", {})
    return None


class MockInspector:
  """Test helper class MockInspector."""

  def inspect(self, _pkg: str) -> dict[str, JsonValue]:
    """Mock implementation of inspect.

    Args:
        _pkg: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {
      "torch.abs": {"name": "abs", "params": ["x"], "docstring_summary": "Calculates abs."},
      "torch.new_thing": {"name": "new_thing", "params": ["a", "b"], "docstring_summary": "Brand new feature."},
    }


def test_matrix_visual_snapshot(snapshot: SnapshotAssert, tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of matrix visual snapshot.

  Args:
      snapshot: Test argument or fixture.
      tmp_path: Test argument or fixture.
  """
  semantics = StableMockSemantics()
  semantics._key_origins = {}
  console = Console(file=None, force_terminal=True, width=100, record=True)
  matrix = CompatibilityMatrix(semantics)
  matrix.console = console
  expected_order: list[str] = ["torch", "jax", "numpy", "tensorflow", "mlx", "paxml"]
  with patch("ml_switcheroo.cli.matrix.get_framework_priority_order", return_value=expected_order):
    matrix.render()
  output: str = console.export_text()

  def header_insensitive(text: str) -> str:
    """Helper to header insensitive.

    Args:
        text: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    lines: list[str] = text.splitlines()
    if not lines:
      return text
    lines[0] = lines[0].strip()
    return "\n".join(lines) + "\n"

  snapshot.assert_match(output, normalizer=header_insensitive)
