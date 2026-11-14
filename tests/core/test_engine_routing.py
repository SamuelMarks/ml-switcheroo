"""Test suite for ASTEngine compiler vs AST rewriter pipeline routing."""

from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture(scope="module")
def real_semantics() -> SemanticsManager:
  """Provide a real ``SemanticsManager`` instance for pipeline routing verification.

  Returns:
      Initialized ``SemanticsManager``.
  """
  return SemanticsManager()


def test_routing_to_compiler_pipeline_target_isa(real_semantics: SemanticsManager) -> None:
  """Verify targeting ``nvidia_sass`` routes through the compiler pipeline and emits SASS.

  Args:
      real_semantics: Shared ``SemanticsManager`` instance.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="nvidia_sass")
  engine = ASTEngine(real_semantics, config)
  src = """def f(x):
  return x
"""
  with (
    patch.object(engine, "_run_compiler_pipeline", wraps=engine._run_compiler_pipeline) as spy_compiler,
    patch.object(engine, "_run_rewriter_pipeline", wraps=engine._run_rewriter_pipeline) as spy_rewriter,
  ):
    res = engine.run(src)
  assert spy_compiler.call_count == 1
  assert spy_rewriter.call_count == 0
  assert res.success is True


def test_routing_to_compiler_pipeline_source_isa(real_semantics: SemanticsManager) -> None:
  """Verify sourcing from ``rdna`` routes through the compiler pipeline to emit Python.

  Args:
      real_semantics: Shared ``SemanticsManager`` instance.
  """
  config = RuntimeConfig(source_framework="rdna", target_framework="torch")
  engine = ASTEngine(real_semantics, config)
  with (
    patch.object(engine, "_run_compiler_pipeline", wraps=engine._run_compiler_pipeline) as spy_compiler,
    patch.object(engine, "_run_rewriter_pipeline", wraps=engine._run_rewriter_pipeline) as spy_rewriter,
  ):
    res = engine.run("v_add_f32 v0, v1, v2")
  assert spy_compiler.call_count == 1
  assert spy_rewriter.call_count == 0
  assert res.success is True
  assert "def " in res.code or "v0" in res.code or res.code != ""


def test_routing_to_rewriter_pipeline_high_level(real_semantics: SemanticsManager) -> None:
  """Verify high-level ``torch -> jax`` conversion routes through the AST rewriter pipeline.

  Args:
      real_semantics: Shared ``SemanticsManager`` instance.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(real_semantics, config)
  src = """import torch
def f(x):
  return torch.abs(x)
"""
  with (
    patch.object(engine, "_run_rewriter_pipeline", wraps=engine._run_rewriter_pipeline) as spy_rewriter,
    patch.object(engine, "_run_compiler_pipeline", wraps=engine._run_compiler_pipeline) as spy_compiler,
  ):
    res = engine.run(src)
  assert spy_rewriter.call_count == 1
  assert spy_compiler.call_count == 0
  assert res.success is True
  assert "jnp.abs" in res.code or "jax.numpy.abs" in res.code


def test_python_frontend_invoked_in_compiler_pipeline() -> None:
  """Verify ``PythonFrontend`` is instantiated with input source in compiler pipeline."""
  sem = MagicMock(spec=SemanticsManager)
  sem.get_framework_config.return_value = {}
  config = RuntimeConfig(source_framework="torch", target_framework="nvidia_sass")
  engine = ASTEngine(sem, config)
  with patch("ml_switcheroo.core.engine.PythonFrontend") as mock_front:
    mock_front.return_value.parse_to_graph.return_value = LogicalGraph()
    with patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend:
      mock_get_backend.return_value = MagicMock()
      res = engine.run("x=1")
      assert mock_front.call_args == (("x=1",),)
      assert res is not None
