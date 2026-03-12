import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
import libcst as cst
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode


def get_tracer_mock():
  m = MagicMock()
  m.export.return_value = []
  return m


def test_rewriter_loopback_sharding_full():
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  # Test jax branch
  engine = ASTEngine(config=cfg, source="torch", target="jax", enable_graph_optimization=True)
  engine.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher") as mock_patcher,
    patch("ml_switcheroo.compiler.backends.python_snippet.PythonSnippetEmitter"),
    patch("ml_switcheroo.compiler.sharding.ShardingInferencePass"),
    patch("ml_switcheroo.compiler.sharding_extractor.ShardingExtractionPass"),
    patch("ml_switcheroo.compiler.fusion.QKVFusionPass"),
    patch("ml_switcheroo.compiler.fusion.QKVDefusionPass"),
    patch("ml_switcheroo.compiler.qwen_fusion.SwiGLUFusionPass"),
    patch("ml_switcheroo.compiler.qwen_fusion.SwiGLUDefusionPass"),
    patch("ml_switcheroo.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass"),
    patch("ml_switcheroo.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass"),
  ):
    mock_differ.return_value.diff.return_value = [1]
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
      mock_extractor.return_value.node_map = {}
      engine._run_rewriter_pipeline("code", get_tracer_mock())

  # Test torch branch
  engine_torch = ASTEngine(config=cfg, source="jax", target="torch", enable_graph_optimization=True)
  engine_torch.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher") as mock_patcher,
    patch("ml_switcheroo.compiler.backends.python_snippet.PythonSnippetEmitter"),
    patch("ml_switcheroo.compiler.sharding.ShardingInferencePass"),
    patch("ml_switcheroo.compiler.sharding_extractor.ShardingExtractionPass"),
    patch("ml_switcheroo.compiler.fusion.QKVFusionPass"),
    patch("ml_switcheroo.compiler.fusion.QKVDefusionPass"),
    patch("ml_switcheroo.compiler.qwen_fusion.SwiGLUFusionPass"),
    patch("ml_switcheroo.compiler.qwen_fusion.SwiGLUDefusionPass"),
    patch("ml_switcheroo.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass"),
    patch("ml_switcheroo.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass"),
  ):
    mock_differ.return_value.diff.return_value = [1]
    with patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor:
      mock_extractor.return_value.graph = LogicalGraph(nodes=[LogicalNode("a", "b")], edges=[])
      mock_extractor.return_value.node_map = {}
      engine_torch._run_rewriter_pipeline("code", get_tracer_mock())
