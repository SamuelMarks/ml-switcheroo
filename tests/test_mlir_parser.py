"""Test module."""

from ml_switcheroo.core.mlir.parser import MlirParser


def test_parser_comprehensive():
  """Test element."""
  mlir = """
    %res1, %res2 = "dialect.op1" @symbol (%arg0 : !type1, %arg1 : !type2)
    """
  parser = MlirParser(mlir)
  node = parser.parse()
  assert node is not None
