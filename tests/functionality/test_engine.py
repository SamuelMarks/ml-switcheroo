"""Test suite for the Engine module."""

from unittest.mock import MagicMock, patch
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.rewriter.patcher import PatchAction


def test_engine_roundtrip():
  """Verifies the behavior of engine roundtrip."""
  engine = ASTEngine(source="torch", target="torch")
  source = "x = 5\nprint(x)"
  result = engine.run(source)
  assert isinstance(result, ConversionResult)
  assert result.success
  assert result.code == source
  assert not result.has_errors


def test_graph_optimization_rewriter_path():
  """Verifies the behavior of graph optimization rewriter path."""
  source_code = "x = conv(x)"
  with (
    patch("ml_switcheroo.core.engine.GraphExtractor") as MockExtractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptimizer,
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as MockDiffer,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher") as MockPatcher,
    patch("ml_switcheroo.core.engine.ingest_code") as MockIngest,
  ):
    fake_tree = MagicMock()
    fake_tree.code = source_code
    fake_tree.visit.return_value = fake_tree
    MockIngest.return_value = fake_tree
    extractor_instance = MockExtractor.return_value
    g_orig = LogicalGraph(nodes=[LogicalNode("n1", "MockOp")])
    extractor_instance.graph = g_orig
    extractor_instance.node_map = {"n1": MagicMock()}
    optimizer_instance = MockOptimizer.return_value
    g_opt = LogicalGraph(nodes=[])
    optimizer_instance.optimize.return_value = g_opt
    differ_instance = MockDiffer.return_value
    differ_instance.diff.return_value = [MagicMock(spec=PatchAction)]
    patcher_instance = MockPatcher.return_value
    cfg = RuntimeConfig(source_framework="torch", target_framework="jax", enable_graph_optimization=True)
    engine = ASTEngine(config=cfg)
    engine.run(source_code)
    MockIngest.assert_called_once()
    fake_tree.visit.assert_any_call(extractor_instance)
    optimizer_instance.optimize.assert_called_once_with(g_orig)
    differ_instance.diff.assert_called_once_with(g_orig, g_opt)
    fake_tree.visit.assert_any_call(patcher_instance)
