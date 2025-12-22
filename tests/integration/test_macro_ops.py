"""
Integration Tests for Composite Operations (Macros) - Feature 08.

Verifies:
1.  **Swish Macro**: `torch.swish(x)` -> `x * jax.nn.sigmoid(x)`.
2.  **Mish Macro**: `torch.mish(x)` -> `x * jax.nn.tanh(jax.nn.softplus(x))`.
3.  **Argument Injection**: `{x}` placeholders are replaced with actual arguments.
4.  **Complex Expressions**: Correct handling of operator precedence and nested calls.
"""

import pytest
import ast
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_CODE = """ 
import torch

def activation(x): 
    s = torch.swish(x) 
    m = torch.mish(x) 
    return s, m
"""

EXPECTED_JAX = """ 
import jax.numpy as jnp
import jax.nn as nn

def activation(x): 
    s = x * nn.sigmoid(x) 
    m = x * jnp.tanh(nn.softplus(x)) 
    return s, m
"""


class MacroSemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    self.import_data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # 1. Swish: x * sigmoid(x)
    # Defined on Torch for matching, JAX target uses Macro
    self._inject("Swish", ["x"], source="torch.swish", target_macro="{x} * jax.nn.sigmoid({x})")

    # 2. Mish: x * tanh(softplus(x))
    self._inject("Mish", ["x"], source="torch.mish", target_macro="{x} * jax.numpy.tanh(jax.nn.softplus({x}))")

  def get_all_rng_methods(self):
    return set()

  def get_framework_config(self, framework):
    return {}

  def _inject(self, name, args, source, target_macro):
    variants = {
      "torch": {"api": source},
      "jax": {
        # API is None or generic if macro is used, macro_template takes precedence
        "macro_template": target_macro
      },
    }

    self.data[name] = {"std_args": args, "variants": variants}
    self._reverse_index[source] = (name, self.data[name])


def test_macro_expansion():
  semantics = MacroSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)

  result = engine.run(SOURCE_CODE)

  assert result.success, f"Failed: {result.errors}"
  code = result.code

  # Verify Swish Expansion
  # x * jax.nn.sigmoid(x)
  # Note: LibCST might format spacing differently, check robustly
  assert "x * jax.nn.sigmoid(x)" in code or "x*jax.nn.sigmoid(x)" in code.replace(" ", "")

  # Verify Mish Expansion
  assert "tanh(jax.nn.softplus(x))" in code


def test_macro_argument_rename():
  """
  Verify that if input argument is named differently (e.g. 'input' vs 'x'),
  normalization correctly maps it to the macro placeholder '{x}'.
  """
  # Setup semantics where source arg is 'input' but standard arg is 'x'
  mgr = SemanticsManager()
  mgr.data = {
    "Swish": {
      "std_args": ["x"],
      "variants": {
        "torch": {"api": "torch.swish", "args": {"x": "input"}},
        "jax": {"macro_template": "{x} * sigmoid({x})"},
      },
    }
  }
  mgr._reverse_index = {"torch.swish": ("Swish", mgr.data["Swish"])}
  mgr._key_origins = {}
  mgr.import_data = {}
  mgr.framework_configs = {}
  mgr._known_rng_methods = set()
  mgr.get_all_rng_methods = lambda: set()
  mgr.get_framework_config = lambda f: {}

  code = "y = torch.swish(input=val)"

  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=mgr, config=config)
  result = engine.run(code)

  assert result.success
  # Macro {x} should be replaced by 'val'
  assert "val * sigmoid(val)" in result.code
