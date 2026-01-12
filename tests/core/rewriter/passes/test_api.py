"""
Tests for the ApiPass.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.passes.api import ApiPass
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.semantics.schema import StructuralTraits


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

    # 2. 'sort' with output adapter
    self._inject(
      "sort",
      ["x"],
      {
        "torch": {"api": "torch.sort"},
        "jax": {"api": "jnp.sort", "output_adapter": "lambda x: x[0]"},
      },
    )

    # 3. 'add_' inplace unroll (Mock plugin)
    # Note: Plugin registration happens outside, but semantics link it
    self._inject(
      "add_",
      ["x", "y"],
      {"torch": {"api": "torch.Tensor.add_"}, "jax": {"requires_plugin": "mock_unroll"}},
    )

    # 4. 'unsupported'
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
  ctx = RewriterContext(semantics, config)

  # Disable functional unwrapping by mocking default traits
  # Default is 'apply', causing test failure if 'layer.apply' is used in input.
  # We clear the trait or set it to something else
  semantics.framework_configs["torch"] = {"traits": {"functional_execution_method": None}}

  def _transform(code):
    module = cst.parse_module(code)
    api_pass = ApiPass()
    return api_pass.transform(module, ctx).code

  return _transform


def test_api_call_rewrite(run_pass):
  """Verify standard call rewrite."""
  code = "y = torch.abs(x)"
  res = run_pass(code)
  assert "jnp.abs(x)" in res


def test_output_adapter_application(run_pass):
  """Verify post-processing logic (adapters)."""
  code = "y = torch.sort(x)"
  res = run_pass(code)
  # Expect: (lambda x: x[0])(jnp.sort(x))
  assert "lambda x: x[0]" in res
  assert "jnp.sort(x)" in res


def test_missing_mapping_strict_failure(run_pass):
  """Verify strict mode error bubbling via EscapeHatch."""
  code = "torch.bad()"
  res = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "No mapping available" in res


def test_assignment_unwrapping_passthrough(run_pass):
  """Verify leave_Assign logic runs."""
  # Since we explicitly disabled functional traits in fixture, this should pass through
  code = "res = layer(x)"
  res = run_pass(code)
  assert "layer(x)" in res

  # Verify apply call is NOT unwrapped
  code2 = "res = layer.apply(v, x)"
  res2 = run_pass(code2)
  assert "layer.apply" in res2


def test_arg_normalization_logic(run_pass):
  """Verify arguments can be renamed/reordered."""
  # We must patch an op with arg maps
  # Creating a temp semantic manager for this specific test
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
  ctx = RewriterContext(mgr, conf)
  api_pass = ApiPass()

  code = "s = torch.sum(x, dim=1)"
  module = cst.parse_module(code)
  res = api_pass.transform(module, ctx).code

  assert "jnp.sum(x, axis=1)" in res
  # 'dim' should be mapped to 'axis' via standard arg 'axis'
