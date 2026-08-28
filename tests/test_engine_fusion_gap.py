"""Test suite for the Engine Fusion Gap module."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.conversion_result import ConversionResult
from unittest.mock import MagicMock
import textwrap


def test_engine_fusion_gap() -> None:
  """Verifies the behavior of engine fusion gap."""
  code: str = textwrap.dedent(
    "\n        import torch.nn as nn\n        class Model(nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.fc = nn.Linear(10, 10)\n            def forward(self, x):\n                return self.fc(x)\n    "
  )
  config: RuntimeConfig = RuntimeConfig(
    source_framework="torch", target_framework="jax", enable_sharding=True, enable_graph_optimization=True
  )
  sm: SemanticsManager = SemanticsManager()
  engine: ASTEngine = ASTEngine(sm, config)
  res: ConversionResult = engine._run_rewriter_pipeline(code, MagicMock())
  assert res is not None
