"""
Integration Tests for Tensor Layout Permutation (Feature 5).

Verifies:
1.  DSL updates allow `layout_map` definition.
2.  Rewriter inspects `layout_map`.
3.  Rewrite injects permutation calls on inputs (Argument wrapping).
4.  Rewrite injects permutation calls on outputs (Result wrapping).
"""

import pytest
import libcst as cst
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier

SOURCE_LAYOUT = """
import torch

def process_image(x):
    # Assume source is NCHW
    return torch.conv2d(x, w)
"""

# Target expectation: JAX uses NHWC sometimes implicitly but let's assume we map to NHWC op
EXPECTED_OUTPUT = """
import jax.numpy as jnp

def process_image(x):
    # Assume source is NCHW
    return jnp.transpose(jax.lax.conv(jnp.transpose(x, axes=(0, 2, 3, 1)), w), axes=(0, 3, 1, 2))
"""


class LayoutSemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    # New attributes
    self._providers = {}
    self._source_registry = {}

    self.import_data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # 1. Define Conv2d with Layout Map
    # Standard: NCHW (implicit)
    # Target (JAX Mock): NHWC required
    self.data["Conv2d"] = {
      "std_args": ["input", "weight"],
      "variants": {
        "torch": {"api": "torch.conv2d"},
        "jax": {
          "api": "jax.lax.conv",
          "args": {"input": "lhs", "weight": "rhs"},
          "layout_map": {
            "input": "NCHW->NHWC",  # (0, 2, 3, 1)
            "return": "NHWC->NCHW",  # (0, 3, 1, 2)
          },
        },
      },
    }
    self._reverse_index["torch.conv2d"] = ("Conv2d", self.data["Conv2d"])

    # 2. Define permute_dims for injection logic
    self.data["permute_dims"] = {
      "std_args": ["x", "axes"],
      "variants": {"jax": {"api": "jnp.transpose", "pack_to_tuple": "axes"}},
    }

  def get_all_rng_methods(self):
    return set()


def test_layout_permutation_injection():
  semantics = LayoutSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)

  result = engine.run(SOURCE_LAYOUT)
  assert result.success
  code = result.code

  # Clean up whitespace
  clean = code.replace(" ", "").replace("\n", "")

  # Verify Input Permutation (NCHW->NHWC: 0, 2, 3, 1)
  assert "jnp.transpose(x,axes=(0,2,3,1))" in clean

  # Verify Output Permutation (NHWC->NCHW: 0, 3, 1, 2)
  # Wrapping logic puts transpose outside
  # jnp.transpose(jax.lax.conv(...), axes=(0, 3, 1, 2))
  assert "jnp.transpose(jax.lax.conv" in clean
  assert ",axes=(0,3,1,2))" in clean

  # Ensure inner call was renamed
  assert "jax.lax.conv" in result.code
