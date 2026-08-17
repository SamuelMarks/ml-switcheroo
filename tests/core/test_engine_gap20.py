"""Tests for engine gap 20."""

from ml_switcheroo.core.engine import ASTEngine
import ml_switcheroo.core.engine
from ml_switcheroo.config import RuntimeConfig


def test_run_compiler_pipeline_rdna_mocked(monkeypatch):
  """Test compiler pipeline with mocked RDNA."""
  import ml_switcheroo.core.engine

  monkeypatch.setattr(ml_switcheroo.core.engine, "is_isa_source", lambda x: True)

  engine = ASTEngine(source="rdna", target="jax")
  monkeypatch.setattr(ml_switcheroo.core.engine, "get_backend_class", lambda x: None)
  result = engine.run("v_add_f32 v0, v1, v2")
  assert not result.success


def test_astengine_fusion_target_branches_mocked(monkeypatch):
  """Test ASTEngine fusion target branches."""
  import ml_switcheroo.core.engine

  class FakeBackend:
    def __init__(self, semantics):
      pass

    def set_mode(self, *args, **kwargs):
      pass

    def compile(self, graph):
      from ml_switcheroo.core.compiler.backends.base import BackendResult

      return BackendResult(code="COMP", imports=["A"], attrs=[])

  monkeypatch.setattr(ml_switcheroo.core.engine, "get_backend_class", lambda x: FakeBackend)

  cfg = RuntimeConfig(source_framework="sass", target_framework="jax", enable_graph_optimization=True)
  engine = ASTEngine(config=cfg)

  engine.target = "flax"
  engine.run("v_add_f32 v0, v1, v2")

  engine.target = "flax_nnx"
  engine.run("v_add_f32 v0, v1, v2")

  engine.target = "paxml"
  engine.run("v_add_f32 v0, v1, v2")

  engine.target = "torch"  # hit else branch
  engine.run("v_add_f32 v0, v1, v2")


def test_astengine_sass_unsupported_target(monkeypatch):
  """Test ASTEngine unsupported target for SASS."""
  monkeypatch.setattr(ml_switcheroo.core.engine, "get_backend_class", lambda x: None)
  engine = ASTEngine(source="sass", target="jax")
  result = engine.run("v_add_f32 v0, v1, v2")
  assert not result.success


def test_astengine_unsupported_isa_frontend(monkeypatch):
  """Test ASTEngine unsupported ISA frontend."""
  engine = ASTEngine(source="torch", target="jax")
  monkeypatch.setattr(ml_switcheroo.core.engine, "is_isa_source", lambda x: True)
  result = engine.run("def a(): pass")
  assert not result.success


def test_astengine_init_branches(monkeypatch):
  """Test ASTEngine init branches."""
  engine1 = ASTEngine(source="torch", target="jax", intermediate="onnx")
  assert engine1.config.intermediate == "onnx"

  cfg1 = RuntimeConfig(source_framework="torch", target_framework="jax", intermediate="mlir")
  engine2 = ASTEngine(config=cfg1)
  assert engine2.config.intermediate == "mlir"

  cfg2 = RuntimeConfig(source_framework="torch", target_framework="jax")
  ASTEngine(config=cfg2)
  assert not cfg2.validation_report


def test_astengine_run_branches(monkeypatch):
  """Test ASTEngine run branches."""
  engine = ASTEngine(source="torch", target="stablehlo")
  code = "def my_func(): pass"
  import libcst as cst

  monkeypatch.setattr(ml_switcheroo.core.engine, "ingest_code", lambda *args: cst.parse_module("def foo(): pass"))

  class MockEmitter:
    def __init__(self, semantics):
      pass

    def convert(self, tree):
      class MockText:
        def to_text(self):
          return "mock"

      return MockText()

  monkeypatch.setattr("ml_switcheroo.core.mlir.stablehlo_emitter.StableHloEmitter", MockEmitter)
  engine.run(code)
