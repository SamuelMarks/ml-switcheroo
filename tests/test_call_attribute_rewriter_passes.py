"""Test suite for Call and Attribute rewriter passes and ImportFixer.

Verifies:
1. Remapping tensor attributes (.shape, .dtype, .ndim, .T, .device).
2. Remapping argument conventions (dim <-> axis, keepdim <-> keepdims).
3. Data type constants (torch.float32 <-> jnp.float32 <-> mx.float32).
4. Import fixing, pruning, and canonical injection.
"""

from ml_switcheroo.core.engine import ASTEngine


def test_remap_argument_conventions_dim_axis() -> None:
  """Tests remapping argument conventions between PyTorch and JAX/MLX/Keras."""
  torch_code: str = """
import torch

def compute(x):
    return torch.sum(x, dim=1, keepdim=True)
"""
  engine_to_jax: ASTEngine = ASTEngine(source="torch", target="jax")
  res_jax = engine_to_jax.run(torch_code)
  assert "axis=1" in res_jax.code
  assert "keepdims=True" in res_jax.code
  assert "jnp.sum" in res_jax.code

  engine_to_mlx: ASTEngine = ASTEngine(source="torch", target="mlx")
  res_mlx = engine_to_mlx.run(torch_code)
  assert "axis=1" in res_mlx.code
  assert "keepdims=True" in res_mlx.code
  assert "mx.sum" in res_mlx.code

  engine_to_keras: ASTEngine = ASTEngine(source="torch", target="keras")
  res_keras = engine_to_keras.run(torch_code)
  assert "axis=1" in res_keras.code
  assert "keepdims=True" in res_keras.code
  assert "keras.ops.sum" in res_keras.code

  # Test reverse: JAX -> PyTorch
  jax_code: str = """
import jax.numpy as jnp

def compute(x):
    return jnp.sum(x, axis=1, keepdims=True)
"""
  engine_to_torch: ASTEngine = ASTEngine(source="jax", target="torch")
  res_torch = engine_to_torch.run(jax_code)
  assert "dim=1" in res_torch.code
  assert "keepdim=True" in res_torch.code
  assert "torch.sum" in res_torch.code


def test_remap_datatype_constants() -> None:
  """Tests remapping data type constants across frameworks."""
  torch_types_code: str = """
import torch

t_f32 = torch.float32
t_f16 = torch.float16
t_f64 = torch.float64
t_i32 = torch.int32
t_i64 = torch.int64
t_bool = torch.bool
"""
  engine_jax: ASTEngine = ASTEngine(source="torch", target="jax")
  res_jax = engine_jax.run(torch_types_code)
  assert "jnp.float32" in res_jax.code
  assert "jnp.float16" in res_jax.code
  assert "jnp.float64" in res_jax.code
  assert "jnp.int32" in res_jax.code
  assert "jnp.int64" in res_jax.code
  assert "jnp.bool_" in res_jax.code

  engine_mlx: ASTEngine = ASTEngine(source="torch", target="mlx")
  res_mlx = engine_mlx.run(torch_types_code)
  assert "mx.float32" in res_mlx.code
  assert "mx.float16" in res_mlx.code
  assert "mx.float64" in res_mlx.code
  assert "mx.int32" in res_mlx.code
  assert "mx.int64" in res_mlx.code
  assert "mx.bool_" in res_mlx.code


def test_remap_tensor_attributes() -> None:
  """Tests preservation and remapping of tensor attributes across frameworks."""
  torch_code: str = """
import torch

def inspect_tensor(x):
    s = x.shape
    d = x.dtype
    n = x.ndim
    t = x.T
    dev = x.device
    return s, d, n, t, dev
"""
  engine_jax: ASTEngine = ASTEngine(source="torch", target="jax")
  res_jax = engine_jax.run(torch_code)
  assert "x.shape" in res_jax.code
  assert "x.dtype" in res_jax.code
  assert "x.ndim" in res_jax.code
  assert "x.T" in res_jax.code
  assert "x.device" in res_jax.code


def test_import_fixing_and_canonical_injection() -> None:
  """Tests removing obsolete imports and injecting canonical target imports."""
  torch_code: str = """
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.val = torch.float32

    def forward(self, x):
        return x
"""
  engine_jax: ASTEngine = ASTEngine(source="torch", target="jax")
  res_jax = engine_jax.run(torch_code)
  assert "import torch" not in res_jax.code
  assert "from torch import nn" not in res_jax.code
  assert "import jax" in res_jax.code

  engine_mlx: ASTEngine = ASTEngine(source="torch", target="mlx")
  res_mlx = engine_mlx.run(torch_code)
  assert "import torch" not in res_mlx.code
  assert "import mlx.nn as nn" in res_mlx.code
