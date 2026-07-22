"""Test suite for the Engine Fusion Gap module."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from unittest.mock import MagicMock
import textwrap


def test_engine_fusion_gap():
  """Verifies the behavior of engine fusion gap."""
  code = textwrap.dedent(
    "\n        import torch.nn as nn\n        class Model(nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.fc = nn.Linear(10, 10)\n            def forward(self, x):\n                return self.fc(x)\n    "
  )
  config = RuntimeConfig(
    source_framework="torch", target_framework="jax", enable_sharding=True, enable_graph_optimization=True
  )
  sm = SemanticsManager()
  engine = ASTEngine(sm, config)
  res = engine._run_rewriter_pipeline(code, MagicMock())
  assert res is not None
