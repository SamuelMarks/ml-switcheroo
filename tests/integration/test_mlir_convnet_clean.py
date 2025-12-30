"""
Integration Test for Clean MLIR Code Generation (ConvNet).
"""

import pytest
import textwrap
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier

# Input code from prompt
INPUT_CODE = """
import torch
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
"""


class MockConvNetSemantics(SemanticsManager):
  def __init__(self):
    # ... Init logic to map torch -> flax_nnx ...
    self.data = {}
    # New attributes
    self._providers = {}
    self._source_registry = {}

    self.import_data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()
    self.test_templates = {}

    self.framework_configs = {
      "flax_nnx": {
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
          "requires_super_init": True,
          # Enable super init to prove it's generated as 'super().__init__()' effectively (Expr)
          # not '_0 = super().__init__()' (Assign)
        },
        "alias": {"module": "flax.nnx", "name": "nnx"},
      },
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward", "requires_super_init": True}},
    }

    # Mappings
    self._add("Conv2d", "torch.nn.Conv2d", "flax.nnx.Conv", ["in", "out", "k"])
    self._add("Linear", "torch.nn.Linear", "flax.nnx.Linear", ["in", "out"])
    # Map flatten to nnx.Flatten as requested by user example structure implication
    self._add("flatten", "torch.flatten", "flax.nnx.Flatten", ["x", "start_dim"])
    self._add("Module", "torch.nn.Module", "flax.nnx.Module", [])

  def get_all_rng_methods(self):
    return set()

  def get_framework_config(self, framework):
    return self.framework_configs.get(framework, {})

  def get_import_map(self, target_fw):
    return {}

  def _add(self, name, s_api, t_api, args):
    variants = {"torch": {"api": s_api}, "flax_nnx": {"api": t_api}}
    self.data[name] = {"std_args": args, "variants": variants}
    self._reverse_index[s_api] = (name, self.data[name])
    # Force neural tier for state injection logic
    self._key_origins[name] = SemanticTier.NEURAL.value


def test_clean_mlir_generation():
  semantics = MockConvNetSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)

  # Enable MLIR intermediate
  engine = ASTEngine(semantics=semantics, config=config, intermediate="mlir")

  result = engine.run(INPUT_CODE)

  assert result.success
  code = result.code

  print("\n[Generated Code via MLIR]:")
  print(code)

  # 1. Assertion: No void assignments for super
  # Check that '_X = super().__init__()' does not exist
  # Only 'super().__init__()' as statement
  import re

  assert not re.search(r"_[a-z0-9_]+\s+=\s+super\(\)\.__init__\(\)", code)
  assert "super().__init__()" in code

  # 2. Assertion: Forward pass is sequential (re-rolled)
  # def __call__(self, x):
  #     _conv_attr = self.conv
  #     _call = _conv_attr(x)
  #     ...
  #     return _linear_call

  # Check for absence of nested call
  # e.g. self.fc(self.conv(x))
  # Or self.fc(torch.flatten(...))

  # 'self.fc(' should exist
  # but the argument inside should be a variable, not a function call
  fc_call_pattern = r"self\.fc\([a-zA-Z_0-9]+\)"
  flatten_call_pattern = r"flax\.nnx\.Flatten\("

  # Ensure flatten call is NOT inside fc call
  # Simple check: line containing self.fc should not contain Flatten
  for line in code.splitlines():
    if "self.fc(" in line:
      assert "Flatten(" not in line, f"Found nested Flatten in: {line}"
      assert "self.conv(" not in line, f"Found nested Conv in: {line}"

  # 3. Check for specific SSA assignments to verify re-rolling happened
  # The output should contain assignments for the intermediate results
  assert " = self.conv" in code  # GetAttr
  assert " = flax.nnx.Flatten" in code or " = nnx.Flatten" in code
