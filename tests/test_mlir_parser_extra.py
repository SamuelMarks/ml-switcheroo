"""Test module."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.mlir.cst import ModuleNode


def test_parse_empty() -> None:
  """Test element."""
  parser: MlirParser = MlirParser("")
  node: ModuleNode = parser.parse()
  assert len(node.body.operations) == 0


def test_parse_unexpected_token() -> None:
  """Test element."""
  parser: MlirParser = MlirParser("~")
  with pytest.raises(ValueError, match="Unexpected"):
    parser.parse()


def test_attribute_alias() -> None:
  """Test element."""
  mlir: str = """
    #map0 = "some_string"
    #arr = [1, 2]
    module {
    }
    """
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  assert len(node.aliases) == 2
  assert node.aliases[0].name == "#map0"
  assert node.aliases[1].name == "#arr"


def test_operation_complex() -> None:
  """Test element."""
  mlir: str = """
    %0 = "foo.bar" (%1, %2) { attr1 = "val1", attr2 = [1, 2, 3] } {
      ^bb0(%arg0: i32, %arg1: f32):
        "foo.yield"() : () -> ()
    } : (i32, i32) -> i32

    "foo.return" %0 : i32

    sw.add %0, %1 : i32
    """
  parser: MlirParser = MlirParser(mlir)
  node: ModuleNode = parser.parse()
  assert len(node.body.operations) == 3
