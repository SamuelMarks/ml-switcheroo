"""Test module."""

from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser
from ml_switcheroo.core.mlir.cst import ModuleNode


def test_stablehlo_parser() -> None:
  """Test element."""
  text: str = "module {}"
  parser: StableHloParser = StableHloParser(text)
  ast: ModuleNode = parser.parse()
  assert isinstance(ast, ModuleNode)
