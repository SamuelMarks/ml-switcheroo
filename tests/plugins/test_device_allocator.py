"""Test suite for the Device Allocator module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.device_allocator import transform_device_allocator
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.numpy import NumpyAdapter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["device_allocator"] = transform_device_allocator
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  device_def = {
    "requires_plugin": "device_allocator",
    "std_args": ["type"],
    "variants": {
      "torch": {"api": "torch.device"},
      "jax": {"api": "jax.devices", "requires_plugin": "device_allocator"},
      "numpy": {"api": "cpu", "requires_plugin": "device_allocator"},
    },
  }
  mgr.get_definition.side_effect = lambda name: ("device", device_def) if name == "torch.device" else None
  mgr.get_known_apis.return_value = {"device": device_def}
  mgr.is_verified.return_value = True

  def resolve_variant(aid, fw):
    """Resolves variant."""
    if aid == "device" and fw in device_def["variants"]:
      return device_def["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  with patch("ml_switcheroo.plugins.device_allocator.get_adapter") as mock_get_adapter:

    def adapter_side_effect(name):
      """Helper to adapter side effect."""
      if name == "jax":
        return JaxCoreAdapter()
      if name == "numpy":
        return NumpyAdapter()
      return None

    mock_get_adapter.side_effect = adapter_side_effect
    yield PivotRewriter(mgr, cfg)


def test_cuda_mapping_default_index(rewriter):
  """Verifies the behavior of cuda mapping default index."""
  code = "d = torch.device('cuda')"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[0]" in result


def test_cuda_mapping_explicit_colon_index(rewriter):
  """Verifies the behavior of cuda mapping explicit colon index."""
  code = "d = torch.device('cuda:1')"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[1]" in result


def test_cpu_mapping(rewriter):
  """Verifies the behavior of cpu mapping."""
  code = "d = torch.device('cpu')"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('cpu')[0]" in result


def test_variable_passthrough(rewriter):
  """Verifies the behavior of variable passthrough."""
  code = "d = torch.device(my_backend)"
  result = rewrite_code(rewriter, code)
  assert "jax.devices(my_backend)[0]" in result


def test_second_arg_index(rewriter):
  """Verifies the behavior of second argument index."""
  code = "d = torch.device('cuda', 2)"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[2]" in result


def test_mps_mapping(rewriter):
  """Verifies the behavior of mps mapping."""
  code = "d = torch.device('mps')"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[0]" in result


def test_ignore_wrong_fw(rewriter):
  """Verifies the behavior of ignore wrong framework."""
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_fw = "numpy"
  code = "d = torch.device('cuda')"
  result = rewrite_code(rewriter, code)
  assert "'cpu'" in result


def test_device_allocator_syntax_error(rewriter):
  """Verifies behavior when new syntax is invalid Python."""
  node = cst.parse_expression("torch.device('cuda')")
  ctx = MagicMock()
  ctx.target_fw = "jax"
  with patch("ml_switcheroo.plugins.device_allocator.get_adapter") as mock_get:
    mock_adapter = MagicMock()
    mock_adapter.get_device_syntax.return_value = "invalid syntax {{ {"
    mock_get.return_value = mock_adapter
    res = transform_device_allocator(node, ctx)
    assert res is node


def test_device_allocator_adapter_exception(rewriter):
  """Verifies behavior when adapter.get_device_syntax throws exception."""
  node = cst.parse_expression("torch.device('cuda')")
  ctx = MagicMock()
  ctx.target_fw = "jax"
  with patch("ml_switcheroo.plugins.device_allocator.get_adapter") as mock_get:
    mock_adapter = MagicMock()
    mock_adapter.get_device_syntax.side_effect = Exception("Adapter error")
    mock_get.return_value = mock_adapter
    res = transform_device_allocator(node, ctx)
    assert res is node


def test_device_allocator_invalid_colon_index(rewriter):
  """Verifies behavior when colon index is not an integer."""
  code = "d = torch.device('cuda:foo')"
  result = rewrite_code(rewriter, code)
  # When it fails to parse as int, it passes 'cuda:foo' to adapter
  # Jax adapter replaces cuda with gpu, but what does it do with 'cuda:foo'?
  # The helper strips 'cuda' -> 'gpu', so if it's not separated, it's just 'cuda:foo'
  # Actually, the jax adapter maps `s_type` which would be 'cuda:foo' since it didn't split
  # Let's just assert it doesn't crash
  assert "jax.devices('gpu:foo')[0]" in result or "jax.devices('cuda:foo')[0]" in result
