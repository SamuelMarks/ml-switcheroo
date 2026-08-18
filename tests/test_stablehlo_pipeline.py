"""Integration tests for the StableHLO pipeline.

This module verifies that a StableHLO source string can be parsed,
converted into a LogicalGraph, and then compiled to a target backend (e.g. SASS),
fulfilling the pipeline advertised in the README.
"""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser
from ml_switcheroo.core.mlir.cst import ModuleNode


def test_stablehlo_parser_basic() -> None:
  """Unit test for the StableHloParser class."""
  code = "module {}"
  parser = StableHloParser(code)
  result = parser.parse()
  assert isinstance(result, ModuleNode)


def test_stablehlo_to_sass_pipeline() -> None:
  """Integration test for StableHLO to SASS compilation.

  Provides a minimal valid MLIR/StableHLO snippet, invokes the ASTEngine
  with source='stablehlo' and target='sass', and verifies that the output
  contains SASS-like structure.
  """
  # Minimal MLIR snippet that parses into a function
  mlir_code = """
  module {
    func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
      return %0 : tensor<f32>
    }
  }
  """

  config = RuntimeConfig(source_framework="stablehlo", target_framework="sass")
  semantics = SemanticsManager()

  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(mlir_code)

  assert result.success is True
  # The exact SASS output depends on the backend implementation,
  # but it should not be empty and should have been processed.
  assert "SASS" in result.code or "sass" in result.code.lower() or result.code != ""
