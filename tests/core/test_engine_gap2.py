"""Test suite for the Engine Gap2 module."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
import libcst as cst
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode


def get_tracer_mock():
  """Gets tracer mock."""
  m = MagicMock()
  m.export.return_value = []
  return m


def test_engine_target_torch_keras_sharding():
  """Verifies the behavior of engine target PyTorch Keras sharding."""
  engine = ASTEngine(source="sass", target="keras")
  engine.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.SassParser"),
    patch("ml_switcheroo.core.engine.SassLifter"),
    patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend,
  ):
    mock_cls = MagicMock()
    mock_cls.__name__ = "PythonBackend"
    mock_get_backend.return_value = mock_cls
    mock_cls.return_value.compile.return_value = "py code"
    engine._run_compiler_pipeline("code", get_tracer_mock())


def test_rewriter_loopback():
  """Verifies the behavior of rewriter loopback."""
  engine = ASTEngine(source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    real_graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
    real_map = {"a": None}
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = real_graph
      mock_extractor.return_value.node_map = real_map
      mock_differ.return_value.diff.return_value = [1]
      with patch("libcst.Module.visit", return_value=cst.parse_module("def bar(): pass")):
        engine._run_rewriter_pipeline("code", get_tracer_mock())


def test_rewriter_loopback_sharding_jax():
  """Verifies the behavior of rewriter loopback sharding JAX."""
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine = ASTEngine(config=cfg, source="torch", target="jax", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    real_graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
    real_map = {"a": None}
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = real_graph
      mock_extractor.return_value.node_map = real_map
      mock_differ.return_value.diff.return_value = False
      with patch("libcst.Module.visit", return_value=cst.parse_module("def bar(): pass")):
        engine._run_rewriter_pipeline("code", get_tracer_mock())


def test_rewriter_loopback_sharding_torch():
  """Verifies the behavior of rewriter loopback sharding PyTorch."""
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  engine = ASTEngine(config=cfg, source="jax", target="torch", enable_graph_optimization=True)
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    real_graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
    real_map = {"a": None}
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = real_graph
      mock_extractor.return_value.node_map = real_map
      mock_differ.return_value.diff.return_value = False
      with patch("libcst.Module.visit", return_value=cst.parse_module("def bar(): pass")):
        engine._run_rewriter_pipeline("code", get_tracer_mock())


def test_engine_target_rdna():
  """Verifies the behavior of engine target RDNA."""
  engine = ASTEngine(source="rdna", target="keras")
  with (
    patch("ml_switcheroo.core.engine.RdnaParser"),
    patch("ml_switcheroo.core.engine.RdnaLifter"),
    patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend,
  ):
    mock_cls = MagicMock()
    mock_cls.__name__ = "PythonBackend"
    mock_get_backend.return_value = mock_cls
    mock_cls.return_value.compile.return_value = "py code"
    engine._run_compiler_pipeline("code", get_tracer_mock())
