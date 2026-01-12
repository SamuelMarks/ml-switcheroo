"""
Tests for Gather Semantics Plugin (Decoupled).

Verifies:
1. Reordering of arguments works if mapping exists.
2. Pass-through logic when mapping is missing (no hardcoded JAX).
3. Keyword handling.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.gather import transform_gather
from ml_switcheroo.frameworks.base import register_framework


def rewrite_code(rewriter, code):
  """Executes pipeline."""
  return rewriter.convert(cst.parse_module(code)).code


@register_framework("custom_fw")
class CustomAdapter:
  @property
  def harness_imports(self):
    return []

  def get_harness_init_code(self):
    return ""

  def get_to_numpy_code(self) -> str:
    """Implement required protocol method."""
    return "return str(obj)"

  @property
  def declared_magic_args(self):
    return []


@pytest.fixture
def rewriter_factory():
  hooks._HOOKS["gather_adapter"] = transform_gather
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define variants
  gather_def = {
    "variants": {
      "torch": {"api": "torch.gather"},
      "jax": {"api": "jnp.take_along_axis", "requires_plugin": "gather_adapter"},
      "custom_fw": {
        "api": "custom.gather_nd",
        "requires_plugin": "gather_adapter",
      },
    }
  }

  # Mock Lookups
  def get_def(name):
    if "gather" in name:
      return ("Gather", gather_def)
    return None

  mgr.get_definition.side_effect = get_def

  # Wiring Logic
  def resolve(aid, fw):
    if aid == "Gather" and fw in gather_def["variants"]:
      return gather_def["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"Gather": gather_def}
  mgr.is_verified.return_value = True

  # Needed for utils.is_framework_module_node to detect torch vs x
  mgr.framework_configs = {"torch": {}, "jax": {}, "custom_fw": {}}

  def create(target):
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_gather_method_reorder_jax(rewriter_factory):
  """
  Input: x.gather(1, indices)
  Output: jnp.take_along_axis(x, indices, 1)
  """
  rw = rewriter_factory("jax")
  code = "y = x.gather(1, indices)"
  res = rewrite_code(rw, code)

  assert "jnp.take_along_axis" in res
  clean = res.replace(" ", "")
  assert "(x,indices,1)" in clean or "(x,indices,1,)" in clean


def test_gather_missing_target_passthrough(rewriter_factory):
  """
  Scenario: Target 'numpy' (not defined in mock semantics).
  Expectation: Return original node (no default to JAX).
  """
  rw = rewriter_factory("numpy")
  # Force context update
  rw.context.hook_context.target_fw = "numpy"

  code = "y = torch.gather(x, 1, idx)"
  res = rewrite_code(rw, code)

  # Should remain torch.gather or return unmodified node which rewriter might keep as-is
  assert "torch.gather" in res
  assert "jnp" not in res
  assert "take_along_axis" not in res


def test_gather_custom_fw_transpilation(rewriter_factory):
  """
  Scenario: Target Custom Framework defined in semantics.
  """
  rw = rewriter_factory("custom_fw")
  code = "y = torch.gather(x, 1, idx)"
  res = rewrite_code(rw, code)

  assert "custom.gather_nd" in res
