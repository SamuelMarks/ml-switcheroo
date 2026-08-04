"""Test module."""

from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.mlir.cst import ModuleNode


def test_mlir_parser_basic():
  """Test for test_mlir_parser_basic."""
  text = """module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
        return %0 : tensor<f32>
      }
    }"""
  parser = MlirParser(text)
  node = parser.parse()
  assert isinstance(node, ModuleNode)
  assert node.to_text() == text
