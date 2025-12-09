"""
Tests for Class Structure Rewriting (Torch -> Flax NNX).

Verifies Feature 050:
1. Inheritance: `torch.nn.Module` -> `flax.nnx.Module`.
2. Constructor: `__init__(self)` -> `__init__(self, rngs: nnx.Rngs)`.
3. Calls: `self.layer(...)` inside init gets `rngs=rngs`.
4. Method: `forward` -> `__call__`.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import StructuralTraits


class MockNNXSemantics(SemanticsManager):
  """
  Mock Manager with Neural Tier definitions mapped for NNX.
  """

  def __init__(self):
    # Initial empty state
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}

    # Configure Traits for JAX to ensure consistent NNX behavior
    self.framework_configs = {
      "jax": {
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
          "requires_super_init": False,
        }
      },
      # FIX: Source framework MUST declare traits so 'torch.nn.Module' is detected
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
    }

    # 1. Define 'Linear' (Neural)
    self._inject("Linear", SemanticTier.NEURAL, "torch", "torch.nn.Linear", "jax", "flax.nnx.Linear")

    # 2. Define 'relu' (Math/Functional)
    self._inject("relu", SemanticTier.ARRAY_API, "torch", "torch.relu", "jax", "flax.nnx.relu")

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(self, name, tier, s_fw, s_api, t_fw, t_api):
    variants = {s_fw: {"api": s_api}, t_fw: {"api": t_api}}
    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = tier.value


@pytest.fixture
def rewriter():
  semantics = MockNNXSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter, code):
  """Executes rewrite and returns source string."""
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_inheritance_rewrite(rewriter):
  """
  Input: class MyModel(torch.nn.Module):
  Output: class MyModel(flax.nnx.Module):
  """
  code = """ 
class MyModel(torch.nn.Module): 
    pass
"""
  result = rewrite_code(rewriter, code)
  assert "class MyModel(flax.nnx.Module):" in result


def test_init_signature_injection(rewriter):
  """
  Input: def __init__(self, features):
  Output: def __init__(self, rngs: flax.nnx.Rngs, features):
  (Argument 1 after self).
  """
  code = """ 
class MyModel(torch.nn.Module): 
    def __init__(self, features): 
        pass
"""
  result = rewrite_code(rewriter, code)

  # Check signature
  # Loose check for argument existence
  assert "def __init__(self, rngs: flax.nnx.Rngs, features):" in result


def test_init_layer_instantiation(rewriter):
  """
  Scenario: Instantiating a Neural Tier object in __init__.
  Input: self.dense = torch.nn.Linear(10, 20)
  Output: self.dense = flax.nnx.Linear(10, 20, rngs=rngs)
  """
  code = """ 
class MyModel(torch.nn.Module): 
    def __init__(self): 
        self.dense = torch.nn.Linear(10, 20) 
"""
  result = rewrite_code(rewriter, code)

  # We expect 'rngs=rngs' to be appended
  assert "flax.nnx.Linear(10, 20, rngs=rngs)" in result


def test_forward_renaming(rewriter):
  """
  Input: def forward(self, x):
  Output: def __call__(self, x):
  """
  code = """ 
class MyModel(torch.nn.Module): 
    def forward(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)

  assert "def __call__(self, x):" in result
  assert "def forward" not in result


def test_non_module_classes_ignored(rewriter):
  """
  Verify that classes NOT inheriting from torch.nn.Module are untouched.
  """
  code = """ 
class DataContainer: 
    def __init__(self, x): 
        self.x = x
    def forward(self): 
        pass
"""
  result = rewrite_code(rewriter, code)

  # Preserved
  assert "class DataContainer:" in result
  assert "def __init__(self, x):" in result  # No rngs injected
  assert "def forward(self):" in result  # No rename


def test_call_site_args_commas(rewriter):
  """
  Ensure comma handling is correct when appending rngs.
  """
  code = """ 
class M(torch.nn.Module): 
    def __init__(self): 
        # No trailing comma
        self.l1 = torch.nn.Linear(1, 2) 
        # Trailing comma
        self.l2 = torch.nn.Linear(3, 4,) 
"""
  result = rewrite_code(rewriter, code)

  # Both should be valid syntax
  assert "flax.nnx.Linear(1, 2, rngs=rngs)" in result
  assert "flax.nnx.Linear(3, 4, rngs=rngs)" in result
