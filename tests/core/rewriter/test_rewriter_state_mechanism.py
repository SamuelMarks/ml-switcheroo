"""Test suite for the Rewriter State Mechanism module."""

import libcst as cst
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockStateSemantics(SemanticsManager):
  """Test double `MockStateSemantics` used in rewriter state mechanism tests."""

  def __init__(self) -> None:
    """Initializes the MockStateSemantics instance."""
    super().__init__(load_defaults=False)
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {
      "tensorflow": {"stateful_call": {"method": "apply", "prepend_arg": "variables"}},
      "mlx": {
        "stateful_call": {"method": "call_fn", "prepend_arg": "ctx"},
        "traits": {
          "inject_magic_args": [["ctx", "custom.Context"]],
          "module_base": "mlx.nn.Module",
        },
      },
      "torch": {"traits": {"module_base": "torch.nn.Module"}},
    }
    self._inject("Linear", SemanticTier.NEURAL, "torch", "torch.Linear", "tensorflow", "func.Dense")
    linear_variants = self.data["Linear"]["variants"]
    assert isinstance(linear_variants, dict)
    linear_variants["mlx"] = {"api": "custom.Layer"}

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, tier: SemanticTier, s_fw: str, s_api: str, t_fw: str, t_api: str) -> None:
    """Execute `_inject` on `MockStateSemantics` for rewriter state mechanism tests.

    Args:
        name: Identifier or operation name string.
        tier: Input `tier` parameter value (`SemanticTier`).
        s_fw: Input `s_fw` parameter value (`str`).
        s_api: Input `s_api` parameter value (`str`).
        t_fw: Input `t_fw` parameter value (`str`).
        t_api: Input `t_api` parameter value (`str`).
    """
    variants: dict[str, JsonValue] = {s_fw: {"api": s_api}, t_fw: {"api": t_api}}
    op_def: dict[str, JsonValue] = {
      "variants": variants,
      "std_args": ["x"],
    }
    self.data[name] = op_def
    self._reverse_index[s_api] = (name, op_def)
    self._key_origins[name] = tier.value

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier (`str`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    defn = self.data.get(abstract_id)
    if not isinstance(defn, dict):
      return None
    variants = defn.get("variants")
    if not isinstance(variants, dict):
      return None
    target = variants.get(fw)
    return target if isinstance(target, dict) else None

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified.

    Args:
        _id: Input `_id` parameter value (`str`).

    Returns:
        Resulting `bool` from is verified.
    """
    return True


@pytest.fixture
def rewriter() -> TestRewriter:
  """Create a configured `TestRewriter` fixture for rewriter state mechanism tests.

  Returns:
      Configured `rewriter` fixture value (TestRewriter).
  """
  semantics = MockStateSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite_code(rewriter: TestRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return str(new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_signature_injection_missing_arg(rewriter: TestRewriter) -> None:
  """Verify `rewrite_code` handles signature injection missing arg as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nclass Net:\n    def __init__(self):\n        self.layer = torch.Linear(10, 10)\n\n    def forward(self, x):\n        return self.layer(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "self.layer.apply(variables, x)" in result


def test_signature_no_injection_if_present(rewriter: TestRewriter) -> None:
  """Verify `rewrite_code` handles signature no injection if present as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nclass Net:\n    def __init__(self):\n        self.layer = torch.Linear(10, 10)\n\n    def forward(self, variables, x):\n        return self.layer(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "self.layer.apply(variables, x)" in result
  assert "Injected missing state argument" not in result


def test_custom_trait_injection() -> None:
  """Verify `RuntimeConfig` handles custom trait injection as expected."""
  semantics = MockStateSemantics()
  config = RuntimeConfig(target_framework="mlx", strict_mode=False)
  custom_rewriter = TestRewriter(semantics, config)
  code: str = "\nclass Net(torch.nn.Module):\n    def __init__(self):\n        self.layer = torch.Linear(10)\n\n    def forward(self, input):\n        return self.layer(input)\n"
  result: str = rewrite_code(custom_rewriter, code)
  assert "def forward(self, ctx, input):" in result or "def forward(self, input):" not in result
  assert "self.layer.call_fn(ctx, input)" in result
