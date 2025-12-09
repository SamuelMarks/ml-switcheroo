"""
Tests for Class Structure Rewriting: PyTorch -> Flax NNX.
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

    # Configure Traits for JAX (Flax NNX)
    self.framework_configs = {
      "jax": {
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
          "requires_super_init": False,
        }
      },
      # FIX: Source Trait
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
    }

    # Define 'Linear' (Neural)
    self._inject(
      "Linear",
      SemanticTier.NEURAL,
      "torch",
      "torch.nn.Linear",
      "jax",
      "flax.nnx.Linear",
    )

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

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
  code = """ 
class MyModel(torch.nn.Module): 
    pass
"""
  result = rewrite_code(rewriter, code)
  assert "class MyModel(flax.nnx.Module):" in result


def test_init_signature_injection(rewriter: PivotRewriter) -> None:
  """
  Verifies `rngs` injection into constructor.
  Expects result to contain `rngs: flax.nnx.Rngs`.
  """
  code = """ 
class MyModel(torch.nn.Module): 
    def __init__(self, features): 
        pass
"""
  result = rewrite_code(rewriter, code)
  assert "def __init__(self, rngs: flax.nnx.Rngs, features):" in result


def test_init_layer_instantiation(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(torch.nn.Module): 
    def __init__(self): 
        self.dense = torch.nn.Linear(10, 20) 
"""
  result = rewrite_code(rewriter, code)
  assert "flax.nnx.Linear(10, 20, rngs=rngs)" in result


def test_forward_renaming(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(torch.nn.Module): 
    def forward(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)
  assert "def __call__(self, x):" in result
  assert "def forward" not in result


def test_non_module_classes_ignored(rewriter: PivotRewriter) -> None:
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


def test_call_site_args_commas(rewriter: PivotRewriter) -> None:
  code = """ 
class M(torch.nn.Module): 
    def __init__(self): 
        # No trailing comma
        self.l1 = torch.nn.Linear(1, 2) 
        # Trailing comma
        self.l2 = torch.nn.Linear(3, 4,) 
"""
  result = rewrite_code(rewriter, code)

  assert "flax.nnx.Linear(1, 2, rngs=rngs)" in result
  assert "flax.nnx.Linear(3, 4, rngs=rngs)" in result
