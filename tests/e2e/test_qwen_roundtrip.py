"""Test suite for the Qwen Roundtrip module."""

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics() -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for qwen roundtrip tests.

  Returns:
      Configured `semantics` fixture value (SemanticsManager).
  """
  return SemanticsManager()


QWEN_SOURCE = "import torch\nimport torch.nn as nn\n\nclass QwenBlock(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.q_proj = nn.Linear(1024, 1024)\n        self.k_proj = nn.Linear(1024, 1024)\n        self.v_proj = nn.Linear(1024, 1024)\n\n        self.gate_proj = nn.Linear(1024, 4096)\n        self.up_proj = nn.Linear(1024, 4096)\n        self.down_proj = nn.Linear(4096, 1024)\n\n    def forward(self, x):\n        q = self.q_proj(x)\n        k = self.k_proj(x)\n        v = self.v_proj(x)\n\n        gate = self.gate_proj(x)\n        up = self.up_proj(x)\n        mlp_out = self.down_proj(gate * up)\n\n        return q, mlp_out\n"


def test_qwen_roundtrip_torch_to_flax_to_torch(semantics: SemanticsManager) -> None:
  """Verify `RuntimeConfig` handles qwen roundtrip torch to flax to torch as expected.

  Args:
      semantics: SemanticsManager instance or mock providing operation mappings.
  """
  config1 = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", enable_graph_optimization=True)
  engine1 = ASTEngine(semantics=semantics, config=config1)
  result1 = engine1.run(QWEN_SOURCE)
  assert result1.success, result1.errors
  flax_code = result1.code
  config2 = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", enable_graph_optimization=True)
  engine2 = ASTEngine(semantics=semantics, config=config2)
  result2 = engine2.run(flax_code)
  assert result2.success, result2.errors
  torch_code_reconstructed = result2.code
  assert "nn.Linear" in torch_code_reconstructed
  assert "forward" in torch_code_reconstructed


def test_qwen_roundtrip_torch_to_mlx_to_keras(semantics: SemanticsManager) -> None:
  """Verify `RuntimeConfig` handles qwen roundtrip torch to MLX to keras as expected.

  Args:
      semantics: SemanticsManager instance or mock providing operation mappings.
  """
  config1 = RuntimeConfig(source_framework="torch", target_framework="mlx", enable_graph_optimization=True)
  engine1 = ASTEngine(semantics=semantics, config=config1)
  result1 = engine1.run(QWEN_SOURCE)
  assert result1.success, result1.errors
  mlx_code = result1.code
  config2 = RuntimeConfig(source_framework="mlx", target_framework="keras", enable_graph_optimization=True)
  engine2 = ASTEngine(semantics=semantics, config=config2)
  result2 = engine2.run(mlx_code)
  assert result2.success, result2.errors
  keras_code = result2.code
  assert "layers.Dense" in keras_code
  assert "call" in keras_code
