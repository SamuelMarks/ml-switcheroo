"""
Tests for the main ASTEngine and result structures.
"""

from unittest.mock import MagicMock, patch

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.rewriter.patcher import PatchAction


def test_engine_roundtrip():
  """Ensure we can parse and emit code without changes (idempotency)."""
  engine = ASTEngine(source="torch", target="torch")
  source = "x = 5\nprint(x)"
  result = engine.run(source)
  assert isinstance(result, ConversionResult)
  assert result.success
  assert result.code == source
  assert not result.has_errors


def test_graph_optimization_rewriter_path():
  """Verify Graph Optimization loopback wiring."""
  source_code = "x = conv(x)"

  # Patch where engine IMPORTS these classes
  with (
    patch("ml_switcheroo.core.engine.GraphExtractor") as MockExtractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptimizer,
    patch("ml_switcheroo.compiler.differ.GraphDiffer") as MockDiffer,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher") as MockPatcher,
    patch("ml_switcheroo.core.engine.ingest_code") as MockIngest,
  ):
    # Setup Mock Tree
    # NOTE: MockIngest must return a Mock object that BEHAVES like CST Node for visit()
    fake_tree = MagicMock()
    fake_tree.code = source_code
    # Allow chaining: tree.visit returns tree
    fake_tree.visit.return_value = fake_tree
    MockIngest.return_value = fake_tree

    # Setup Extractor
    extractor_instance = MockExtractor.return_value
    g_orig = LogicalGraph(nodes=[LogicalNode("n1", "MockOp")])
    extractor_instance.graph = g_orig
    extractor_instance.node_map = {"n1": MagicMock()}

    # Setup Optimizer
    optimizer_instance = MockOptimizer.return_value
    g_opt = LogicalGraph(nodes=[])
    optimizer_instance.optimize.return_value = g_opt

    # Setup Differ
    differ_instance = MockDiffer.return_value
    differ_instance.diff.return_value = [MagicMock(spec=PatchAction)]

    # Setup Patcher
    patcher_instance = MockPatcher.return_value

    # Run
    cfg = RuntimeConfig(source_framework="torch", target_framework="jax", enable_graph_optimization=True)
    engine = ASTEngine(config=cfg)
    engine.run(source_code)

    # Assert
    MockIngest.assert_called_once()
    fake_tree.visit.assert_any_call(extractor_instance)
    optimizer_instance.optimize.assert_called_once_with(g_orig)
    differ_instance.diff.assert_called_once_with(g_orig, g_opt)
    fake_tree.visit.assert_any_call(patcher_instance)
