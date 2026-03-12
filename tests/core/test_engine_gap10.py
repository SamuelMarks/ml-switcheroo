import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from unittest.mock import patch, MagicMock


def test_engine_strict_mode_linter_errors():
  config = RuntimeConfig(strict_mode=True)
  engine = ASTEngine(source="jax", target="torch", config=config)
  code = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"

  with patch("ml_switcheroo.core.engine.StructuralLinter.check") as MockCheck:
    MockCheck.return_value = ["linter error"]
    with patch("ml_switcheroo.core.engine.ingest_code") as MockIngest:
      mock_tree = MagicMock()
      mock_tree.code = "import jax"

      with patch("ml_switcheroo.core.engine.RewriterPipeline") as MockPipeCls:
        MockPipeCls.return_value.run.return_value = mock_tree
        with patch("ml_switcheroo.core.engine.ImportFixer") as MockFixerCls:
          mock_fixer_visit = MagicMock()
          mock_fixer_visit.code = "import jax"
          mock_tree.visit.return_value = mock_fixer_visit

          MockIngest.return_value = mock_tree
          engine.config.enable_graph_optimization = False
          res = engine._run_rewriter_pipeline(code, MagicMock())
          assert "linter error" in res.errors[0]


def test_engine_escape_hatches_detected():
  config = RuntimeConfig()
  engine = ASTEngine(source="jax", target="torch", config=config)
  code = "import jax.numpy as jnp\nx = jnp.array([1, 2])\n"

  from ml_switcheroo.core.escape_hatch import EscapeHatch

  with patch("ml_switcheroo.core.engine.ingest_code") as MockIngest:
    mock_tree = MagicMock()
    mock_tree.code = f"{EscapeHatch.START_MARKER} some code"

    with patch("ml_switcheroo.core.engine.RewriterPipeline") as MockPipeCls:
      MockPipeCls.return_value.run.return_value = mock_tree
      with patch("ml_switcheroo.core.engine.ImportFixer") as MockFixerCls:
        mock_fixer_visit = MagicMock()
        mock_fixer_visit.code = mock_tree.code
        mock_tree.visit.return_value = mock_fixer_visit

        MockIngest.return_value = mock_tree
        engine.config.enable_graph_optimization = False
        res = engine._run_rewriter_pipeline(code, MagicMock())
        assert "Escape Hatches Detected" in res.errors[0]


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

          with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFrontend:
            MockFrontend.return_value.parse_to_graph.return_value = MagicMock(nodes=["n1"])
            engine._run_compiler_pipeline(code, MagicMock())
            MockSharding.assert_called()
