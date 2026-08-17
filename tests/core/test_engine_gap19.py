"""Tests for engine gap 19."""

from ml_switcheroo.core.engine import ASTEngine


def test_astengine_stablehlo_branch(monkeypatch):
  """Test ASTEngine stablehlo branch."""
  # Test the branch elif self.target == "stablehlo" without mocking ingest_code
  engine = ASTEngine(source="torch", target="stablehlo")
  # Provide simple valid python code
  code = "def f(): pass"

  # We mock StableHloEmitter so we don't need its implementation
  class MockEmitter:
    def __init__(self, semantics):
      pass

    def convert(self, tree):
      class TextObj:
        def to_text(self):
          return "mlir"

      return TextObj()

  monkeypatch.setattr("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter", MockEmitter)
  # also we need to avoid the 'self.target' check in some places maybe?
  result = engine.run(code)
  assert result.success
  assert result.code == "mlir"
