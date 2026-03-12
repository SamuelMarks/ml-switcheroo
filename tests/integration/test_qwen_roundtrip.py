import pytest
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics():
  return SemanticsManager()


QWEN_SOURCE = """import torch
import torch.nn as nn

class QwenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(1024, 1024)
        self.k_proj = nn.Linear(1024, 1024)
        self.v_proj = nn.Linear(1024, 1024)
        
        self.gate_proj = nn.Linear(1024, 4096)
        self.up_proj = nn.Linear(1024, 4096)
        self.down_proj = nn.Linear(4096, 1024)
        
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        mlp_out = self.down_proj(gate * up) 
        
        return q, mlp_out
"""


def test_qwen_roundtrip_torch_to_flax_to_torch(semantics):
  # 1. Torch -> Flax NNX
  config1 = RuntimeConfig(
    source_framework="torch",
    target_framework="flax_nnx",
    enable_graph_optimization=True,
  )
  engine1 = ASTEngine(semantics=semantics, config=config1)
  result1 = engine1.run(QWEN_SOURCE)
  assert result1.success, result1.error

  flax_code = result1.code

  # 2. Flax NNX -> Torch
  config2 = RuntimeConfig(
    source_framework="flax_nnx",
    target_framework="torch",
    enable_graph_optimization=True,
  )
  engine2 = ASTEngine(semantics=semantics, config=config2)
  result2 = engine2.run(flax_code)

  assert result2.success, result2.error
  torch_code_reconstructed = result2.code

  # Assert structural integrity
  assert "nn.Linear" in torch_code_reconstructed
  assert "forward" in torch_code_reconstructed


def test_qwen_roundtrip_torch_to_mlx_to_keras(semantics):
  # 1. Torch -> MLX
  config1 = RuntimeConfig(
    source_framework="torch",
    target_framework="mlx",
    enable_graph_optimization=True,
  )
  engine1 = ASTEngine(semantics=semantics, config=config1)
  result1 = engine1.run(QWEN_SOURCE)
  assert result1.success, result1.error

  mlx_code = result1.code

  # 2. MLX -> Keras
  config2 = RuntimeConfig(
    source_framework="mlx",
    target_framework="keras",
    enable_graph_optimization=True,
  )
  engine2 = ASTEngine(semantics=semantics, config=config2)
  result2 = engine2.run(mlx_code)

  assert result2.success, result2.error
  keras_code = result2.code

  assert "layers.Dense" in keras_code
  assert "call" in keras_code
