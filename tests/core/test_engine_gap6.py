import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from unittest.mock import patch, MagicMock


def test_engine_target_torch_sharding_rewriter_3():
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"

  with patch("ml_switcheroo.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.compiler.sharding_extractor.ShardingExtractionPass.apply") as MockExtPass:
      with patch("ml_switcheroo.compiler.differ.GraphDiffer.diff", return_value=None):
        with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
          MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
          with patch("ml_switcheroo.core.graph.GraphExtractor") as MockExtCls:
            MockExt = MockExtCls.return_value
            g = MagicMock()
            g.nodes = ["n1"]
            MockExt.graph = g
            MockExt.node_map = {}

            engine._run_rewriter_pipeline(code, MagicMock())
            MockSharding.assert_called()
