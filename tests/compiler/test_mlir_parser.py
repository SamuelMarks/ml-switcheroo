"""Test module."""

import typing

from ml_switcheroo.core.mlir.cst import ModuleNode
from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_basic() -> None:
  """Docstring."""
  text: str = """module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
        return %0 : tensor<f32>
      }
    }"""
  parser = MlirParser(text)
  node: typing.Any = parser.parse()
  assert isinstance(node, ModuleNode)
  assert node.to_text() == text
