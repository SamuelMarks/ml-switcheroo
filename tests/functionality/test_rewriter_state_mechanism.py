"""Test suite for the Rewriter State Mechanism module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo_ir.schema.ghost import SemanticTier


class MockStateSemantics(SemanticsManager):
  """Mock State Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockStateSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {
      "tensorflow": {"stateful_call": {"method": "apply", "prepend_arg": "variables"}},
      "mlx": {
        "stateful_call": {"method": "call_fn", "prepend_arg": "ctx"},
        "traits": {"inject_magic_args": [("ctx", "custom.Context")], "module_base": "mlx.nn.Module"},
      },
      "torch": {"traits": {"module_base": "torch.nn.Module"}},
    }
    self._inject("Linear", SemanticTier.NEURAL, "torch", "torch.Linear", "tensorflow", "func.Dense")
    self.data["Linear"]["variants"]["mlx"] = {"api": "custom.Layer"}

  def get_framework_config(self, framework: str):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, tier: SemanticTier, s_fw: str, s_api: str, t_fw: str, t_api: str) -> None:
    """Mock implementation of  inject."""
    variants = {s_fw: {"api": s_api}, t_fw: {"api": t_api}}
    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = tier.value


@pytest.fixture
def rewriter() -> TestRewriter:
  """Provides a mock rewriter for testing."""
  semantics = MockStateSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite_code(rewriter: TestRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_signature_injection_missing_arg(rewriter: TestRewriter) -> None:
  """Verifies the behavior of signature injection missing argument."""
  code = "\nclass Net:\n    def __init__(self):\n        self.layer = torch.Linear(10, 10)\n\n    def forward(self, x):\n        return self.layer(x)\n"
  result = rewrite_code(rewriter, code)
  assert "self.layer.apply(variables, x)" in result


def test_signature_no_injection_if_present(rewriter: TestRewriter) -> None:
  """Verifies the behavior of signature no injection if present."""
  code = "\nclass Net:\n    def __init__(self):\n        self.layer = torch.Linear(10, 10)\n\n    def forward(self, variables, x):\n        return self.layer(x)\n"
  result = rewrite_code(rewriter, code)
  assert "self.layer.apply(variables, x)" in result
  assert "Injected missing state argument" not in result


def test_custom_trait_injection() -> None:
  """Verifies the behavior of custom trait injection."""
  semantics = MockStateSemantics()
  config = RuntimeConfig(target_framework="mlx", strict_mode=False)
  custom_rewriter = TestRewriter(semantics, config)
  code = "\nclass Net(torch.nn.Module):\n    def __init__(self):\n        self.layer = torch.Linear(10)\n\n    def forward(self, input):\n        return self.layer(input)\n"
  result = rewrite_code(custom_rewriter, code)
  assert "def forward(self, ctx, input):" in result or "def forward(self, input):" not in result
  assert "self.layer.call_fn(ctx, input)" in result
