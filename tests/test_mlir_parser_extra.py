"""Test module."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_parse_empty():
  """Test element."""
  parser = MlirParser("")
  node = parser.parse()
  assert len(node.body.operations) == 0


def test_parse_unexpected_token():
  """Test element."""
  parser = MlirParser("~")
  with pytest.raises(ValueError, match="Unexpected"):
    parser.parse()


def test_attribute_alias():
  """Test element."""
  mlir = """
    #map0 = "some_string"
    #arr = [1, 2]
    module {
    }
    """
  parser = MlirParser(mlir)
  node = parser.parse()
  assert len(node.aliases) == 2
  assert node.aliases[0].name == "#map0"
  assert node.aliases[1].name == "#arr"


def test_operation_complex():
  """Test element."""
  mlir = """
    %0 = "foo.bar" (%1, %2) { attr1 = "val1", attr2 = [1, 2, 3] } {
      ^bb0(%arg0: i32, %arg1: f32):
        "foo.yield"() : () -> ()
    } : (i32, i32) -> i32

    "foo.return" %0 : i32

    sw.add %0, %1 : i32
    """
  parser = MlirParser(mlir)
  node = parser.parse()
  assert len(node.body.operations) == 3
