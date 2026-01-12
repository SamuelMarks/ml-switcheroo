"""
Tests for Generic State Mechanism handling.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
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

    self.framework_configs = {
      "tensorflow": {"stateful_call": {"method": "apply", "prepend_arg": "variables"}},
      "mlx": {
        "stateful_call": {"method": "call_fn", "prepend_arg": "ctx"},
        # FIX: Add traits to enable arg injection in StructuralPass
        "traits": {
          "inject_magic_args": [("ctx", "custom.Context")],
          "module_base": "mlx.nn.Module",
        },
      },
      # FIX: Add torch configuration so source class is detected
      "torch": {"traits": {"module_base": "torch.nn.Module"}},
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
def rewriter() -> TestRewriter:
  """Rewriter defaulting to 'tensorflow' target (simulating functional)."""
  semantics = MockStateSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite_code(rewriter: TestRewriter, code: str) -> str:
  """Executes rewrite pipeline."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_signature_injection_missing_arg(rewriter: TestRewriter) -> None:
  code = """
class Net:
    def __init__(self):
        self.layer = torch.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)
"""
  result = rewrite_code(rewriter, code)
  assert "self.layer.apply(variables, x)" in result


def test_signature_no_injection_if_present(rewriter: TestRewriter) -> None:
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
  # Using TestRewriter shim
  custom_rewriter = TestRewriter(semantics, config)

  # Note: 'func' is NOT a standard inference method (forward/call),
  # so standard StructuralPass might skip injection unless we add it to known methods.
  # However, for this test, we verify the CALL rewrite mostly. But the assertion checks signature.
  # Let's use 'forward' to trigger the injection reliably.
  code = """
class Net(torch.nn.Module):
    def __init__(self):
        self.layer = torch.Linear(10)

    def forward(self, input):
        return self.layer(input)
"""
  result = rewrite_code(custom_rewriter, code)

  # Expect 'ctx' injection and 'call_fn' methods based on config
  assert "def forward(self, ctx, input):" in result or "def forward(self, input):" not in result
  assert "self.layer.call_fn(ctx, input)" in result
