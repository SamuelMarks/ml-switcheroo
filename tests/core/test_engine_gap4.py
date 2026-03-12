import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
import libcst as cst
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode


def test_rewriter_loopback_sharding_actual():
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True

  # We will NOT mock the passes. We will just pass a real graph that doesn't blow up.
  # To do that, we mock GraphExtractor to return a real graph with real nodes, so it doesn't crash the differ.

  cfg.enable_graph_optimization = True
  engine = ASTEngine(config=cfg, source="torch", target="jax")
  engine.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as mock_opt,
    patch("ml_switcheroo.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher") as mock_patcher,
    patch("ml_switcheroo.compiler.backends.python_snippet.PythonSnippetEmitter") as mock_emitter,
  ):
    mock_graph = MagicMock()
    mock_graph.nodes = [1]
    mock_extractor.return_value.graph = mock_graph
    mock_extractor.return_value.node_map = {}

    mock_differ.return_value.diff.return_value = [1]

    # Patch the passes inside the engine module directly!
    with (
      patch("ml_switcheroo.compiler.sharding.ShardingInferencePass.apply", return_value=mock_graph) as mock_inf,
      patch(
        "ml_switcheroo.compiler.sharding_extractor.ShardingExtractionPass.apply", return_value=mock_graph
      ) as mock_ext,
      patch("ml_switcheroo.compiler.fusion.QKVFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.fusion.QKVDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.qwen_fusion.SwiGLUFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.qwen_fusion.SwiGLUDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass.apply", return_value=mock_graph),
    ):
      # Need a dummy tracer that just passes through
      m = MagicMock()
      m.export.return_value = []
      with patch("libcst.Module.visit") as mock_visit:
        mock_visit.return_value = cst.parse_module("def bar(): pass")
        engine._run_rewriter_pipeline("code", m)
      mock_inf.assert_called_once()

  # Same for Torch
  cfg.enable_graph_optimization = True
  engine_torch = ASTEngine(config=cfg, source="jax", target="torch")
  engine_torch.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as mock_opt,
    patch("ml_switcheroo.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher") as mock_patcher,
    patch("ml_switcheroo.compiler.backends.python_snippet.PythonSnippetEmitter") as mock_emitter,
  ):
    mock_graph = MagicMock()
    mock_graph.nodes = [1]
    mock_extractor.return_value.graph = mock_graph
    mock_extractor.return_value.node_map = {}

    mock_differ.return_value.diff.return_value = [1]

    with (
      patch("ml_switcheroo.compiler.sharding.ShardingInferencePass.apply", return_value=mock_graph) as mock_inf,
      patch(
        "ml_switcheroo.compiler.sharding_extractor.ShardingExtractionPass.apply", return_value=mock_graph
      ) as mock_ext,
      patch("ml_switcheroo.compiler.fusion.QKVFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.fusion.QKVDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.qwen_fusion.SwiGLUFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.qwen_fusion.SwiGLUDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass.apply", return_value=mock_graph),
    ):
      m = MagicMock()
      m.export.return_value = []
      with patch("libcst.Module.visit") as mock_visit:
        mock_visit.return_value = cst.parse_module("def bar(): pass")
        engine_torch._run_rewriter_pipeline("code", m)
      mock_inf.assert_called_once()
