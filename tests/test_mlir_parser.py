"""Test module."""

from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.mlir.cst import ModuleNode


def test_parser_comprehensive() -> None:
  """Test element."""
  mlir: str = """
    %res1, %res2 = "dialect.op1" @symbol (%arg0 : !type1, %arg1 : !type2)
    """
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  assert node is not None
