"""
Tests for State Container Plugin (Torch -> Match JAX/NNX idioms).
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
  new_tree = tree.visit(rewriter)
  return new_tree.code


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

  # Mock definitions allowing loose matching on method name
  def get_definition_side_effect(name):
    if "register_buffer" in name:
      return "reg_buf", {"variants": {"jax": {"requires_plugin": "torch_register_buffer_to_nnx"}}}
    if "register_parameter" in name:
      return "reg_param", {"variants": {"jax": {"requires_plugin": "torch_register_parameter_to_nnx"}}}
    if "state_dict" in name and "load" not in name:
      return "state_dict", {"variants": {"jax": {"requires_plugin": "torch_state_dict_to_nnx"}}}
    if "load_state_dict" in name:
      return "load_state_dict", {"variants": {"jax": {"requires_plugin": "torch_load_state_dict_to_nnx"}}}
    if "parameters" in name and "register" not in name:
      return "parameters", {"variants": {"jax": {"requires_plugin": "torch_parameters_to_nnx"}}}
    return None

  mgr.get_definition.side_effect = get_definition_side_effect

  # Mock variant resolution
  def resolve_variant_side_effect(aid, fw):
    if fw != "jax":
      return None
    if aid == "reg_buf":
      return {"requires_plugin": "torch_register_buffer_to_nnx"}
    if aid == "reg_param":
      return {"requires_plugin": "torch_register_parameter_to_nnx"}
    if aid == "state_dict":
      return {"requires_plugin": "torch_state_dict_to_nnx"}
    if aid == "load_state_dict":
      return {"requires_plugin": "torch_load_state_dict_to_nnx"}
    if aid == "parameters":
      return {"requires_plugin": "torch_parameters_to_nnx"}
    return None

  mgr.resolve_variant.side_effect = resolve_variant_side_effect

  # Always Verified
  mgr.is_verified.return_value = True

  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  return PivotRewriter(mgr, config)


def test_register_buffer_transform(rewriter):
  """
  Input: self.register_buffer('running_mean', torch.zeros(10))
  Output: setattr(self, 'running_mean', flax.nnx.BatchStat(torch.zeros(10)))
  """
  code = "self.register_buffer('running_mean', torch.zeros(10))"
  res = rewrite_code(rewriter, code)

  assert "setattr(self, 'running_mean', flax.nnx.BatchStat(torch.zeros(10)))" in res


def test_register_parameter_transform(rewriter):
  """
  Input: self.register_parameter('weight', w)
  Output: setattr(self, 'weight', flax.nnx.Param(w))
  """
  code = "self.register_parameter('weight', w)"
  res = rewrite_code(rewriter, code)

  assert "setattr(self, 'weight', flax.nnx.Param(w))" in res


def test_state_dict_transform(rewriter):
  """
  Input: sd = model.state_dict()
  Output: sd = flax.nnx.state(model).to_pure_dict()
  """
  code = "sd = model.state_dict()"
  res = rewrite_code(rewriter, code)

  assert "flax.nnx.state(model).to_pure_dict()" in res


def test_load_state_dict_transform(rewriter):
  """
  Input: model.load_state_dict(sd)
  Output: flax.nnx.update(model, sd)
  """
  code = "model.load_state_dict(sd)"
  res = rewrite_code(rewriter, code)

  assert "flax.nnx.update(model, sd)" in res


def test_parameters_transform(rewriter):
  """
  Input: p_list = model.parameters()
  Output: p_list = flax.nnx.state(model, flax.nnx.Param).values()
  """
  # Use assignment instead of Loop to avoid triggering LoopUnroll escape hatch
  code = "p_list = model.parameters()"
  res = rewrite_code(rewriter, code)

  assert "flax.nnx.state(model, flax.nnx.Param).values()" in res


def test_nontarget_passthrough(rewriter):
  """Verify no change if target is not JAX."""
  rewriter.ctx._runtime_config.target_framework = "torch"
  rewriter.ctx.target_fw = "torch"
  # Ensure resolve returns None for torch
  rewriter.semantics.resolve_variant.side_effect = lambda aid, fw: None

  code = "model.state_dict()"
  res = rewrite_code(rewriter, code)
  assert "max.state" not in res
  assert "model.state_dict()" in res
