"""Test suite for mlir stablehlo parser."""

from ml_switcheroo.core.mlir.cst import ModuleNode
from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser


def test_stablehlo_parser() -> None:
  """Verify `StableHloParser` handles stablehlo parser as expected."""
  text: str = "module {}"
  parser: StableHloParser = StableHloParser(text)
  ast: ModuleNode = parser.parse()
  assert isinstance(ast, ModuleNode)
  assert ast.to_text() == "module"
