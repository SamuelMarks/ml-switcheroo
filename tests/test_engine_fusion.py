"""Test suite for the Engine Fusion module."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.conversion_result import ConversionResult
import textwrap


def test_engine_fusion_jax() -> None:
  """Verifies the behavior of engine fusion JAX."""
  code: str = textwrap.dedent(
    "\n        class Model(nn.Module):\n            def __init__(self):\n                super().__init__()\n            def forward(self, x):\n                return x\n    "
  )
  config: RuntimeConfig = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    target_flavour="linen",
    enable_sharding=True,
    enable_graph_optimizer=True,
  )
  sm: SemanticsManager = SemanticsManager()
  engine: ASTEngine = ASTEngine(sm, config)
  res: ConversionResult = engine.run(code)
  assert res.success
