"""PyTorch Example Snippets."""

from typing import Dict


def get_torch_tiered_examples() -> Dict[str, str]:
  """Provides code snippets for "Wizard" or "Demo" usage."""
  return {
    "tier1_math": """import torch

def math_ops(x, y):
    # Tier 1: Core Tensor Operations
    a = torch.abs(x)
    b = torch.add(a, y)

    # Reduction
    return torch.mean(b)
""",
    "tier2_neural_simple": """import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        return nn.functional.relu(x)
""",
    "tier2_neural_cnn": """import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
""",
    "tier3_extras_dataloader": """import torch
from torch.utils.data import DataLoader, TensorDataset

def create_loader(data, targets):
    # Tier 3: Data Loader
    ds = TensorDataset(data, targets)
    return DataLoader(ds, batch_size=32, num_workers=4)
""",
    "tier4_qwen3": """import torch
import torch.nn as nn

class QwenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard HF-style separate projections
        self.q_proj = nn.Linear(1024, 1024)
        self.k_proj = nn.Linear(1024, 1024)
        self.v_proj = nn.Linear(1024, 1024)

        self.gate_proj = nn.Linear(1024, 4096)
        self.up_proj = nn.Linear(1024, 4096)
        self.down_proj = nn.Linear(4096, 1024)

    def forward(self, x):
        # Attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # SwiGLU MLP
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Note: switcheroo handles the fusion with SwiGLU
        mlp_out = self.down_proj(gate * up)

        return q, mlp_out
""",
    "tier4_qwen3-vl": """import torch
import torch.nn as nn

class Qwen3VLVisionConfig:
    in_channels: int = 3
    hidden_size: int = 1280
    temporal_patch_size: int = 2
    patch_size: int = 14

class Qwen3VLPatchEmbed(nn.Module):
    '''3D Convolutional patch embedding for vision input.'''
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.config = config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            padding=0,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        seq_len = hidden_states.shape[0]

        hidden_states = hidden_states.reshape(
            seq_len, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size
        )

        out = self.proj(hidden_states)

        return out.reshape(seq_len, cfg.hidden_size)
""",
  }
