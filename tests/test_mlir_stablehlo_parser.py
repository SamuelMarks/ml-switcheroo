"""Test module."""

from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser
from ml_switcheroo.core.mlir.cst import ModuleNode


def test_stablehlo_parser():
  """Test element."""
  text = "module {}"
  parser = StableHloParser(text)
  ast = parser.parse()
  assert isinstance(ast, ModuleNode)
