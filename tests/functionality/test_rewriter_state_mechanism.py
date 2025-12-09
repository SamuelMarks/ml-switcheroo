"""
Tests for Generic State Mechanism handling in PivotRewriter.

Mock updated to support get_framework_config.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import SemanticTier


class MockStateSemantics(SemanticsManager):
  """
  Mock Manager with arbitrary state configurations.
  """

  def __init__(self) -> None:
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}

    # Map 'tensorflow' to represent 'functional_fw' with explicit state passing
    # Map 'mlx' to represent 'custom_fw' with context passing
    self.framework_configs = {
      "tensorflow": {"stateful_call": {"method": "apply", "prepend_arg": "variables"}},
      "mlx": {"stateful_call": {"method": "call_fn", "prepend_arg": "ctx"}},
    }

    # Define a 'Linear' operation that is considered stateful (Tier=Neural)
    self._inject(
      "Linear",
      SemanticTier.NEURAL,
      "torch",
      "torch.Linear",
      "tensorflow",
      "func.Dense",
    )

    # Add variant for MLX
    self.data["Linear"]["variants"]["mlx"] = {"api": "custom.Layer"}

  # --- FIX: Added method ---
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
  """Rewriter defaulting to 'tensorflow' target (simulating functional)."""
  semantics = MockStateSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow", strict_mode=False)
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Executes rewrite."""
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


# ... (Original Tests remain unchanged as the fix is in the Mock class above) ...
def test_signature_injection_missing_arg(rewriter: PivotRewriter) -> None:
  code = """
class Net:
    def __init__(self):
        self.layer = torch.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)
"""
  result = rewrite_code(rewriter, code)
  assert "self.layer.apply(variables, x)" in result


def test_signature_no_injection_if_present(rewriter: PivotRewriter) -> None:
  code = """
class Net:
    def __init__(self):
        self.layer = torch.Linear(10, 10)

    def forward(self, variables, x):
        return self.layer(x)
"""
  result = rewrite_code(rewriter, code)
  assert "self.layer.apply(variables, x)" in result
  assert "Injected missing state argument" not in result


def test_custom_trait_injection() -> None:
  semantics = MockStateSemantics()
  config = RuntimeConfig(target_framework="mlx", strict_mode=False)
  custom_rewriter = PivotRewriter(semantics, config)

  code = """
class Net:
    def __init__(self):
        self.layer = torch.Linear(10)
    
    def func(self, input):
        return self.layer(input)
"""
  result = rewrite_code(custom_rewriter, code)

  # Expect 'ctx' injection and 'call_fn' methods based on config
  assert "def func(self, ctx, input):" in result
  assert "self.layer.call_fn(ctx, input)" in result


def test_injection_outside_method(rewriter: PivotRewriter) -> None:
  code = """
layer = torch.Linear(10)
def run_model(x):
    return layer(x)
"""
  result = rewrite_code(rewriter, code)
  assert "def run_model(variables, x):" in result
  assert "layer.apply(variables, x)" in result
