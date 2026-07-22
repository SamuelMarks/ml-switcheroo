"""Test suite for the Gather module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.gather import transform_gather
from ml_switcheroo.frameworks.base import register_framework


def rewrite_code(rewriter, code):
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@register_framework("custom_fw")
class CustomAdapter:
  """Test suite for the Custom Adapter component."""

  @property
  def harness_imports(self):
    """Helper to harness imports."""
    return []

  def get_harness_init_code(self):
    """Gets harness initialization code."""
    return ""

  def get_to_numpy_code(self) -> str:
    """Gets to NumPy code."""
    return "return str(obj)"

  @property
  def declared_magic_args(self):
    """Helper to declared magic arguments."""
    return []


@pytest.fixture
def rewriter_factory():
  """Provides a mock rewriter factory for testing."""
  hooks._HOOKS["gather_adapter"] = transform_gather
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  gather_def = {
    "variants": {
      "torch": {"api": "torch.gather"},
      "jax": {"api": "jnp.take_along_axis", "requires_plugin": "gather_adapter"},
      "custom_fw": {"api": "custom.gather_nd", "requires_plugin": "gather_adapter"},
    }
  }

  def get_def(name):
    """Gets def."""
    if "gather" in name:
      return ("Gather", gather_def)
    return None

  mgr.get_definition.side_effect = get_def

  def resolve(aid, fw):
    """Resolves ."""
    if aid == "Gather" and fw in gather_def["variants"]:
      return gather_def["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"Gather": gather_def}
  mgr.is_verified.return_value = True
  mgr.framework_configs = {"torch": {}, "jax": {}, "custom_fw": {}}

  def create(target):
    """Creates ."""
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_gather_method_reorder_jax(rewriter_factory):
  """Verifies the behavior of gather method reorder JAX."""
  rw = rewriter_factory("jax")
  code = "y = x.gather(1, indices)"
  res = rewrite_code(rw, code)
  assert "jnp.take_along_axis" in res
  clean = res.replace(" ", "")
  assert "(x,indices,1)" in clean or "(x,indices,1,)" in clean


def test_gather_missing_target_passthrough(rewriter_factory):
  """Verifies the behavior of gather missing target passthrough."""
  rw = rewriter_factory("numpy")
  rw.context.hook_context.target_fw = "numpy"
  code = "y = torch.gather(x, 1, idx)"
  res = rewrite_code(rw, code)
  assert "torch.gather" in res
  assert "jnp" not in res
  assert "take_along_axis" not in res


def test_gather_custom_fw_transpilation(rewriter_factory):
  """Verifies the behavior of gather custom framework transpilation."""
  rw = rewriter_factory("custom_fw")
  code = "y = torch.gather(x, 1, idx)"
  res = rewrite_code(rw, code)
  assert "custom.gather_nd" in res
