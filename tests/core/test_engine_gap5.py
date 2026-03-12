import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from unittest.mock import patch, MagicMock


def test_engine_target_torch_sharding_compiler():
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"

  with patch("ml_switcheroo.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.compiler.sharding_extractor.ShardingExtractionPass.apply") as MockExtPass:
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          engine._run_compiler_pipeline(code, MagicMock())
          MockSharding.assert_called()


def test_engine_target_flax_sharding_compiler():
  config = RuntimeConfig(enable_sharding=True, enable_graph_optimization=True)
  engine = ASTEngine(source="jax", target="flax", config=config)
  code = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"

  with patch("ml_switcheroo.compiler.sharding.ShardingInferencePass.apply") as MockSharding:
    with patch("ml_switcheroo.compiler.sharding_extractor.ShardingExtractionPass.apply") as MockExtPass:
      with patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer") as MockOptCls:
        MockOptCls.return_value.optimize.return_value = MagicMock(nodes=["n1"])
        with patch("ml_switcheroo.core.engine.get_backend_class") as MockGetBackend:
          mock_backend = MagicMock()
          mock_backend.compile.return_value = "print('hi')"
          MockGetBackend.return_value = MagicMock(return_value=mock_backend)
          MockGetBackend.return_value.__name__ = "PythonBackend"
          engine._run_compiler_pipeline(code, MagicMock())
          MockSharding.assert_called()


def test_engine_target_torch_sharding_rewriter_2():
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

            with patch("ml_switcheroo.compiler.backends.python_snippet.PythonSnippetEmitter"):
              with patch("ml_switcheroo.core.rewriter.patcher.GraphPatcher"):
                engine._run_rewriter_pipeline(code, MagicMock())
                MockSharding.assert_called()
