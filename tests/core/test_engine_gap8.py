"""Test suite for the Engine Gap8 module."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from unittest.mock import patch, MagicMock


def test_engine_target_keras_sharding_compiler():
  """Verifies the behavior of engine target Keras sharding compiler."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="keras", config=config)
  code = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFrontend:
            MockFrontend.return_value.parse_to_graph.return_value = MagicMock()
            engine._run_compiler_pipeline(code, MagicMock())
            MockSharding.assert_called()


def test_engine_target_keras_sharding_rewriter():
  """Verifies the behavior of engine target Keras sharding rewriter."""
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="keras", config=config)
  code = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"
  with patch("ml_switcheroo.core.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.core.compiler.sharding_extractor.ShardingExtractionPass.apply"):
      with patch("ml_switcheroo.core.compiler.differ.GraphDiffer.diff", return_value=None):
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
