"""Tests for missing engine code coverage."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from unittest import mock


def test_astengine_init_missing_branches():
  """Test ASTEngine initialization missing branches."""
  engine1 = ASTEngine(source="torch", target="jax", intermediate="onnx")
  assert engine1.config.intermediate == "onnx"

  cfg1 = RuntimeConfig(source_framework="torch", target_framework="jax", intermediate="mlir")
  engine2 = ASTEngine(config=cfg1)
  assert engine2.config.intermediate == "mlir"

  cfg2 = RuntimeConfig(source_framework="torch", target_framework="jax")
  ASTEngine(config=cfg2)
  assert not cfg2.validation_report


def test_astengine_stablehlo_missing_branches(monkeypatch):
  """Test ASTEngine stablehlo missing branches."""
  import ml_switcheroo.core.engine

  engine = ASTEngine(source="torch", target="stablehlo")
  import libcst as cst

  monkeypatch.setattr(ml_switcheroo.core.engine, "ingest_code", lambda *args: cst.parse_module("def foo(): pass"))

  class MockEmitter:
    def __init__(self, semantics):
      pass

    def convert(self, tree):
      class TextObj:
        def to_text(self):
          return "mlir"

      return TextObj()

  monkeypatch.setattr("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter", MockEmitter)
  monkeypatch.setattr("ml_switcheroo.core.engine.get_adapter", lambda *args: None)
  result = engine.run("def f(): pass")
  assert result.success


def test_astengine_sass_unsupported_missing_branches(monkeypatch):
  """Test ASTEngine sass unsupported target missing branches."""
  import ml_switcheroo.core.engine

  # To bypass pydantic validation, we just create the engine normally and override self.target
  engine = ASTEngine(source="sass", target="jax")
  engine.target = "unsupported"

  # We must mock get_backend_class so it hits the check and raises ValueError? Wait, if we mock it to return None
  monkeypatch.setattr(ml_switcheroo.core.engine, "is_isa_source", lambda x: True)
  monkeypatch.setattr(ml_switcheroo.core.engine, "get_backend_class", lambda x: None)

  with (
    mock.patch("ml_switcheroo.core.compiler.frontends.sass.SassParser") as mock_parser,
    mock.patch("ml_switcheroo.core.compiler.frontends.sass.SassLifter") as mock_lifter,
  ):
    mock_parser.return_value.parse.return_value.statements = []
    mock_lifter.return_value.lift.return_value = mock.MagicMock()
    result = engine.run("sass code")
    assert not result.success
    assert any("No backend found" in e for e in result.errors)
