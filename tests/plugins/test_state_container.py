"""
Tests for State Container Plugin.

Verifies:
1. Strict Decoupling: Ensuring transformations abort if semantics are missing.
2. Structure Logic: Ensuring wrappers are constructed accurately when mapped.
3. Component Coverage: Register Buffer, Parameter, State Dict, Load State, Parameters.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.state_container import (
  convert_register_buffer,
  convert_register_parameter,
  convert_state_dict,
  convert_load_state_dict,
  convert_parameters,
)


def rewrite_code(rewriter, code: str) -> str:
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


@pytest.fixture
def rewriter():
  # Explicitly register hooks to bypass dynamic loading issues in test env
  hooks._HOOKS["torch_register_buffer_to_nnx"] = convert_register_buffer
  hooks._HOOKS["torch_register_parameter_to_nnx"] = convert_register_parameter
  hooks._HOOKS["torch_state_dict_to_nnx"] = convert_state_dict
  hooks._HOOKS["torch_load_state_dict_to_nnx"] = convert_load_state_dict
  hooks._HOOKS["torch_parameters_to_nnx"] = convert_parameters
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define mock responses for get_definition
  # This allows the Rewriter to find the Hook trigger
  def get_definition_side_effect(name):
    # Precise matching logic to prevent greedy substring overlaps
    # e.g. "load_state_dict" would match "state_dict" in a naive lookup

    plugin_name = None
    key = None

    if "register_buffer" in name:
      plugin_name = "torch_register_buffer_to_nnx"
      key = "register_buffer"
    elif "register_parameter" in name:
      plugin_name = "torch_register_parameter_to_nnx"
      key = "register_parameter"
    elif "load_state_dict" in name:
      plugin_name = "torch_load_state_dict_to_nnx"
      key = "load_state_dict"
    elif "state_dict" in name:
      plugin_name = "torch_state_dict_to_nnx"
      key = "state_dict"
    elif "parameters" in name:
      plugin_name = "torch_parameters_to_nnx"
      key = "parameters"

    if plugin_name:
      return (
        key,
        {"variants": {"jax": {"requires_plugin": plugin_name}}},
      )
    return None

  mgr.get_definition.side_effect = get_definition_side_effect

  # Define mock responses for resolve_variant
  # This allows the Rewriter API confirmation logic to pass
  def resolve_variant_side_effect(aid, fw):
    if fw != "jax":
      return None
    # Return generic match
    return {"requires_plugin": f"torch_{aid}_to_nnx"}

  mgr.resolve_variant.side_effect = resolve_variant_side_effect
  mgr.is_verified.return_value = True

  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  # Ensure get_framework_config doesn't crash attributes logic
  mgr.get_framework_config.return_value = {}

  return PivotRewriter(mgr, config)


# --- Lookup Helpers for Tests ---
# We inject these behaviors into the HookContext mock lookup_api per test case


def configure_context(rewriter, mapping):
  """Updates the rewriter context lookup behavior."""

  def lookup(name):
    return mapping.get(name)

  rewriter.ctx.lookup_api = MagicMock(side_effect=lookup)


# --- 1. Successful Transformations (Mappings Exist) ---


def test_register_buffer_success(rewriter):
  configure_context(rewriter, {"BatchStat": "flax.nnx.BatchStat"})
  code = "self.register_buffer('running_mean', torch.zeros(10))"
  res = rewrite_code(rewriter, code)
  assert "setattr(self, 'running_mean', flax.nnx.BatchStat(torch.zeros(10)))" in res


def test_register_parameter_success(rewriter):
  configure_context(rewriter, {"Param": "custom.Parameter"})
  code = "self.register_parameter('weight', w)"
  res = rewrite_code(rewriter, code)
  assert "setattr(self, 'weight', custom.Parameter(w))" in res


def test_state_dict_success(rewriter):
  configure_context(rewriter, {"ModuleState": "flax.nnx.state"})
  code = "sd = model.state_dict()"
  res = rewrite_code(rewriter, code)
  assert "flax.nnx.state(model).to_pure_dict()" in res


def test_load_state_dict_success(rewriter):
  configure_context(rewriter, {"UpdateState": "flax.nnx.update"})
  code = "model.load_state_dict(sd)"
  res = rewrite_code(rewriter, code)
  assert "flax.nnx.update(model, sd)" in res


def test_parameters_success(rewriter):
  configure_context(rewriter, {"ModuleState": "state", "Param": "Param"})
  code = "p = model.parameters()"
  res = rewrite_code(rewriter, code)
  assert "state(model, Param).values()" in res


# --- 2. Fallback / Failure Modes (Missing Mappings) ---


def test_register_buffer_missing_mapping(rewriter):
  """Verify abort if BatchStat not defined."""
  configure_context(rewriter, {})  # Empty map
  code = "self.register_buffer('n', t)"
  res = rewrite_code(rewriter, code)
  # Should contain original
  assert "self.register_buffer('n', t)" in res
  assert "setattr" not in res


def test_register_parameter_missing_mapping(rewriter):
  """Verify abort if Param not defined."""
  configure_context(rewriter, {})
  code = "self.register_parameter('n', p)"
  res = rewrite_code(rewriter, code)
  assert "self.register_parameter" in res


def test_state_dict_missing_mapping(rewriter):
  """Verify abort if ModuleState not defined."""
  configure_context(rewriter, {})
  code = "model.state_dict()"
  res = rewrite_code(rewriter, code)
  assert "model.state_dict()" in res


def test_load_state_dict_missing_mapping(rewriter):
  """Verify abort if UpdateState not defined."""
  configure_context(rewriter, {})
  code = "model.load_state_dict(state)"
  res = rewrite_code(rewriter, code)
  assert "model.load_state_dict" in res


def test_parameters_missing_mapping(rewriter):
  """Verify abort if either Param or ModuleState is missing."""
  # Case 1: Both missing
  configure_context(rewriter, {})
  code = "model.parameters()"
  assert "model.parameters()" in rewrite_code(rewriter, code)

  # Case 2: Only Param missing
  configure_context(rewriter, {"ModuleState": "state"})
  assert "model.parameters()" in rewrite_code(rewriter, code)

  # Case 3: Only State missing
  configure_context(rewriter, {"Param": "Param"})
  assert "model.parameters()" in rewrite_code(rewriter, code)
