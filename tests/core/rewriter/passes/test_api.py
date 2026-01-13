"""
Tests for the ApiPass via TestRewriter.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    self.framework_configs = {}
    self.import_data = {}
    self._reverse_index = {}
    self._validation_status = {}

    # 1. 'abs'
    self._inject(
      "abs",
      ["x"],
      {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}},
    )

    # 2. 'add_' inplace unroll (Mock plugin)
    self._inject(
      "add_",
      ["x", "y"],
      {"torch": {"api": "torch.Tensor.add_"}, "jax": {"requires_plugin": "mock_unroll"}},
    )

    # 3. 'unsupported'
    self._inject(
      "unsupported",
      [],
      {"torch": {"api": "torch.bad"}},
    )

    self.framework_configs["jax"] = {"alias": {"module": "jax.numpy", "name": "jnp"}}

  def _inject(self, name, args, variants):
    self.data[name] = {"std_args": args, "variants": variants}
    for _, v in variants.items():
      if "api" in v:
        self._reverse_index[v["api"]] = (name, self.data[name])

  def get_definition(self, name):
    return self._reverse_index.get(name)

  def resolve_variant(self, aid, fw):
    return self.data.get(aid, {}).get("variants", {}).get(fw)

  def is_verified(self, _id):
    return True

  def get_framework_config(self, fw):
    return self.framework_configs.get(fw, {})


@pytest.fixture
def run_pass():
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)

  # Disable functional unwrapping to prevent test noise
  semantics.framework_configs["torch"] = {"traits": {"functional_execution_method": None}}

  rewriter = TestRewriter(semantics, config)

  def _transform(code):
    tree = cst.parse_module(code)
    return rewriter.convert(tree).code

  return _transform


def test_api_call_rewrite(run_pass):
  """Verify standard call rewrite."""
  code = "y = torch.abs(x)"
  res = run_pass(code)
  assert "jnp.abs(x)" in res


def test_missing_mapping_strict_failure(run_pass):
  """Verify strict mode error bubbling via EscapeHatch."""
  code = "torch.bad()"
  res = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "No mapping available" in res


def test_assignment_unwrapping_passthrough(run_pass):
  """Verify leave_Assign logic runs."""
  code = "res = layer(x)"
  res = run_pass(code)
  assert "layer(x)" in res

  # Verify apply call is NOT unwrapped
  code2 = "res = layer.apply(v, x)"
  res2 = run_pass(code2)
  assert "layer.apply" in res2


def test_arg_normalization_logic(run_pass):
  """Verify arguments can be renamed/reordered."""
  mgr = MagicMock(spec=SemanticsManager)
  # Define 'sum' -> std: [x, axis]
  op_def = {
    "std_args": ["x", "axis"],
    "variants": {
      "torch": {"api": "torch.sum", "args": {"axis": "dim"}},
      "jax": {"api": "jnp.sum"},
    },
  }
  mgr.get_definition.return_value = ("Sum", op_def)
  mgr.resolve_variant.return_value = op_def["variants"]["jax"]
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}

  conf = RuntimeConfig(source_framework="torch", target_framework="jax")
  rewriter = TestRewriter(mgr, conf)

  code = "s = torch.sum(x, dim=1)"
  tree = cst.parse_module(code)
  res = rewriter.convert(tree).code

  assert "jnp.sum(x, axis=1)" in res
