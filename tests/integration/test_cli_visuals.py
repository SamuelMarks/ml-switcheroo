"""Test suite for the Cli Visuals module."""

import typing
from unittest.mock import patch

from rich.console import Console

from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.semantics.manager import SemanticsManager


class StableMockSemantics(SemanticsManager):
  """Docstring."""

  def get_known_apis(self) -> dict[str, typing.Any]:
    """Mock implementation of get known apis."""
    return {
      "abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}},
      "magic_op": {
        "std_args": ["x"],
        "variants": {"torch": {"api": "torch.magic"}, "jax": {"requires_plugin": "magic_fix"}},
      },
      "unsupported_op": {"std_args": ["x"], "variants": {"torch": {"api": "torch.oops"}}},
    }

  def get_definition(self, api_name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    if api_name == "torch.abs":
      return ("abs", {})
    return None


class MockInspector:
  """Docstring."""

  def inspect(self, _pkg: str) -> dict[str, typing.Any]:
    """Mock implementation of inspect."""
    return {
      "torch.abs": {"name": "abs", "params": ["x"], "docstring_summary": "Calculates abs."},
      "torch.new_thing": {"name": "new_thing", "params": ["a", "b"], "docstring_summary": "Brand new feature."},
    }


def test_matrix_visual_snapshot(snapshot: typing.Any, tmp_path: typing.Any) -> None:
  """Verifies the behavior of matrix visual snapshot."""
  semantics = StableMockSemantics()
  semantics._key_origins = {}
  console = Console(file=None, force_terminal=True, width=100, record=True)  # type: ignore
  matrix = CompatibilityMatrix(semantics)
  matrix.console = console
  expected_order: list[str] = ["torch", "jax", "numpy", "tensorflow", "mlx", "paxml"]
  with patch("ml_switcheroo.cli.matrix.get_framework_priority_order", return_value=expected_order):
    matrix.render()
  output: str = console.export_text()

  def header_insensitive(text: str) -> str:
    """Helper to header insensitive."""
    lines: list[str] = text.splitlines()
    if not lines:
      return text
    lines[0] = lines[0].strip()
    return "\n".join(lines) + "\n"

  snapshot.assert_match(output, normalizer=header_insensitive)
