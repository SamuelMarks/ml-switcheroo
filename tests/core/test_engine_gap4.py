"""Test suite for the Engine Gap4 module."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
import libcst as cst


def test_rewriter_loopback_sharding_actual():
  """Verifies the behavior of rewriter loopback sharding actual."""
  cfg = RuntimeConfig(strict_mode=False)
  cfg.enable_sharding = True
  cfg.enable_graph_optimization = True
  engine = ASTEngine(config=cfg, source="torch", target="jax")
  engine.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    mock_graph = MagicMock()
    mock_graph.nodes = [1]
    mock_extractor.return_value.graph = mock_graph
    mock_extractor.return_value.node_map = {}
    mock_differ.return_value.diff.return_value = [1]
    with (
      patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply", return_value=mock_graph) as mock_inf,
      patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass.apply", return_value=mock_graph),
    ):
      m = MagicMock()
      m.export.return_value = []
      with patch("libcst.Module.visit") as mock_visit:
        mock_visit.return_value = cst.parse_module("def bar(): pass")
        engine._run_rewriter_pipeline("code", m)
      mock_inf.assert_called_once()
  cfg.enable_graph_optimization = True
  engine_torch = ASTEngine(config=cfg, source="jax", target="torch")
  engine_torch.config.enable_sharding = True
  with (
    patch("ml_switcheroo.core.engine.ingest_code", return_value=cst.parse_module("def foo(): pass")),
    patch("ml_switcheroo.core.engine.GraphExtractor") as mock_extractor,
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer"),
    patch("ml_switcheroo.core.compiler.differ.GraphDiffer") as mock_differ,
    patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"),
    patch("ml_switcheroo.core.compiler.backends.python_snippet.PythonSnippetEmitter"),
  ):
    mock_graph = MagicMock()
    mock_graph.nodes = [1]
    mock_extractor.return_value.graph = mock_graph
    mock_extractor.return_value.node_map = {}
    mock_differ.return_value.diff.return_value = [1]
    with (
      patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply", return_value=mock_graph) as mock_inf,
      patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.fusion.QKVFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.fusion.QKVDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.SwiGLUDefusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingFusionPass.apply", return_value=mock_graph),
      patch("ml_switcheroo.core.compiler.qwen_fusion.VisionPatchEmbeddingDefusionPass.apply", return_value=mock_graph),
    ):
      m = MagicMock()
      m.export.return_value = []
      with patch("libcst.Module.visit") as mock_visit:
        mock_visit.return_value = cst.parse_module("def bar(): pass")
        engine_torch._run_rewriter_pipeline("code", m)
      mock_inf.assert_called_once()
