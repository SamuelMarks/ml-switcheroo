"""Tests for engine gap 19."""

import typing
import pytest
from ml_switcheroo.core.engine import ASTEngine, ConversionResult


def test_astengine_stablehlo_branch(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test ASTEngine stablehlo branch."""
  # Test the branch elif self.target == "stablehlo" without mocking ingest_code
  engine = ASTEngine(source="torch", target="stablehlo")
  # Provide simple valid python code
  code: str = "def f(): pass"

  # We mock StableHloEmitter so we don't need its implementation
  class MockEmitter:
    """Docstring."""

    def __init__(self, semantics: typing.Any) -> None:
      """Docstring."""
      pass

    def convert(self, tree: typing.Any) -> typing.Any:
      """Docstring."""

      class TextObj:
        """Docstring."""

        def to_text(self) -> str:
          """Docstring."""
          return "mlir"

      return TextObj()

  monkeypatch.setattr("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter", MockEmitter)
  # also we need to avoid the 'self.target' check in some places maybe?
  result: ConversionResult = engine.run(code)
  assert result.success
  assert result.code == "mlir"
