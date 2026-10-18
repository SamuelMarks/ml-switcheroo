"""Test suite for the Engine module."""

import typing

from unittest.mock import MagicMock, patch

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.rewriter.patcher import PatchAction


def test_engine_roundtrip() -> None:
  """Verifies the behavior of engine roundtrip."""
  engine: typing.Any = ASTEngine(source="torch", target="torch")
  source: typing.Any = "x = 5\nprint(x)"
  result: typing.Any = engine.run(source)
  assert isinstance(result, ConversionResult)
  assert result.success
  assert result.code == source
  assert not result.has_errors


def test_graph_optimization_rewriter_path() -> None:
  """Verifies the behavior of graph optimization rewriter path."""
  source_code: typing.Any = "x = conv(x)"
  with (
    patch("ml_switcheroo.core.engine.GraphExtractor") as MockExtractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptimizer,
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as MockDiffer,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher") as MockPatcher,
    patch("ml_switcheroo.core.engine.ingest_code") as MockIngest,
  ):
    fake_tree: typing.Any = MagicMock()
    fake_tree.code = source_code
    fake_tree.visit.return_value = fake_tree
    MockIngest.return_value = fake_tree
    extractor_instance: typing.Any = MockExtractor.return_value
    g_orig: typing.Any = LogicalGraph(nodes={n.id: n for n in [LogicalNode("n1", "MockOp")]})
    extractor_instance.graph = g_orig
    extractor_instance.node_map = {"n1": MagicMock()}
    optimizer_instance: typing.Any = MockOptimizer.return_value
    g_opt: typing.Any = LogicalGraph(nodes={})
    optimizer_instance.optimize.return_value = g_opt
    differ_instance: typing.Any = MockDiffer.return_value
    differ_instance.diff.return_value = [MagicMock(spec=PatchAction)]
    patcher_instance: typing.Any = MockPatcher.return_value
    cfg: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax", enable_graph_optimization=True)
    engine: typing.Any = ASTEngine(config=cfg)
    engine.run(source_code)
    MockIngest.assert_called_once()
    fake_tree.visit.assert_any_call(extractor_instance)
    optimizer_instance.optimize.assert_called_once_with(g_orig)
    differ_instance.diff.assert_called_once_with(g_orig, g_opt)
    fake_tree.visit.assert_any_call(patcher_instance)
