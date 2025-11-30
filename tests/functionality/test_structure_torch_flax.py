"""
Tests for Class Structure Rewriting: PyTorch -> Flax NNX.

This module verifies the structural transformation logic required to convert
PyTorch Object-Oriented models into Flax NNX models.

Verifies:
    1.  Inheritance: `torch.nn.Module` -> `flax.nnx.Module`.
    2.  State Injection: `rngs` added to `__init__` signature.
    3.  State Propagation: `rngs` passed to sub-layer definitions.
    4.  Method Renaming: `forward` -> `__call__`.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import SemanticTier


class MockTorchToFlaxSemantics(SemanticsManager):
  """
  Mock Manager providing PyTorch -> JAX mappings.
  """

  def __init__(self) -> None:
    """Initializes mock data."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}

    # Define 'Linear' (Neural)
    self._inject(
      "Linear",
      SemanticTier.NEURAL,
      "torch",
      "torch.nn.Linear",
      "jax",
      "flax.nnx.Linear",
    )

  def _inject(
    self,
    name: str,
    tier: SemanticTier,
    s_fw: str,
    s_api: str,
    t_fw: str,
    t_api: str,
  ) -> None:
    variants = {s_fw: {"api": s_api}, t_fw: {"api": t_api}}
    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = tier.value


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Creates a Rewriter configured for Torch -> JAX translation."""
  semantics = MockTorchToFlaxSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Helper to parse and rewrite code."""
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_inheritance_rewrite(rewriter: PivotRewriter) -> None:
  """
  Verifies module inheritance swap.

  Input: `class MyModel(torch.nn.Module):`
  Output: `class MyModel(flax.nnx.Module):`
  """
  code = """
class MyModel(torch.nn.Module):
    pass
"""
  result = rewrite_code(rewriter, code)
  assert "class MyModel(flax.nnx.Module):" in result


def test_init_signature_injection(rewriter: PivotRewriter) -> None:
  """
  Verifies `rngs` injection into constructor.

  Input: `def __init__(self, features):`
  Output: `def __init__(self, rngs: flax.nnx.Rngs, features):`
  """
  code = """
class MyModel(torch.nn.Module):
    def __init__(self, features):
        pass
"""
  result = rewrite_code(rewriter, code)
  # Loose check for argument existence at position 1
  assert "def __init__(self, rngs: flax.nnx.Rngs, features):" in result


def test_init_layer_instantiation(rewriter: PivotRewriter) -> None:
  """
  Verifies `rngs` propagation to layers in `__init__`.

  Input: `self.dense = torch.nn.Linear(10, 20)`
  Output: `self.dense = flax.nnx.Linear(10, 20, rngs=rngs)`
  """
  code = """
class MyModel(torch.nn.Module):
    def __init__(self):
        self.dense = torch.nn.Linear(10, 20)
"""
  result = rewrite_code(rewriter, code)
  assert "flax.nnx.Linear(10, 20, rngs=rngs)" in result


def test_forward_renaming(rewriter: PivotRewriter) -> None:
  """
  Verifies `forward` to `__call__` renaming.

  Input: `def forward(self, x):`
  Output: `def __call__(self, x):`
  """
  code = """
class MyModel(torch.nn.Module):
    def forward(self, x):
        return x
"""
  result = rewrite_code(rewriter, code)
  assert "def __call__(self, x):" in result
  assert "def forward" not in result


def test_non_module_classes_ignored(rewriter: PivotRewriter) -> None:
  """
  Verifies that classes NOT inheriting from torch.nn.Module are untouched.
  """
  code = """
class DataContainer:
    def __init__(self, x):
        self.x = x
    def forward(self):
        pass
"""
  result = rewrite_code(rewriter, code)
  assert "class DataContainer:" in result
  assert "def __init__(self, x):" in result  # No rngs injected
  assert "def forward(self):" in result  # No rename
