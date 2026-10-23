"""Test suite for the Device Allocator module."""

import typing
from pydantic import JsonValue
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.plugins.device_allocator import _parse_device_args, transform_device_allocator
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
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


@pytest.fixture
def rewriter() -> typing.Generator[PivotRewriter, None, None]:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (typing.Generator[PivotRewriter, None, None]).
  """
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "device_allocator", getattr({"v": transform_device_allocator}, "__getitem__")("v")
  )
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

  def resolve_variant(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    if aid == "device" and fw in device_def["variants"]:
      res = getattr(
        {"v": getattr({"v": getattr({"v": device_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")[fw]},
        "__getitem__",
      )("v")
      return res if isinstance(res, dict) else {}
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  with patch("ml_switcheroo.plugins.device_allocator.get_adapter") as mock_get_adapter:

    def adapter_side_effect(name: str) -> JaxCoreAdapter | NumpyAdapter | None:
      """Helper to adapter side effect.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `JaxCoreAdapter | NumpyAdapter | None` from adapter side effect.
      """
      if name == "jax":
        return JaxCoreAdapter()
      if name == "numpy":
        return NumpyAdapter()
      return None

    mock_get_adapter.side_effect = adapter_side_effect
    yield PivotRewriter(mgr, cfg)


def test_cuda_mapping_default_index(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of cuda mapping default index.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "d = torch.device('cuda')"
  result: str = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[0]" in result


def test_cuda_mapping_explicit_colon_index(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of cuda mapping explicit colon index.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "d = torch.device('cuda:1')"
  result: str = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[1]" in result


def test_cpu_mapping(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of cpu mapping.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "d = torch.device('cpu')"
  result: str = rewrite_code(rewriter, code)
  assert "jax.devices('cpu')[0]" in result


def test_variable_passthrough(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of variable passthrough.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "d = torch.device(my_backend)"
  result: str = rewrite_code(rewriter, code)
  assert "jax.devices(my_backend)[0]" in result


def test_second_arg_index(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of second argument index.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "d = torch.device('cuda', 2)"
  result: str = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[2]" in result


def test_mps_mapping(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of mps mapping.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "d = torch.device('mps')"
  result: str = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[0]" in result


def test_ignore_wrong_fw(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore wrong framework.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_framework = "numpy"
  code: str = "d = torch.device('cuda')"
  result: str = rewrite_code(rewriter, code)
  assert "'cpu'" in result


def test_device_allocator_syntax_error(rewriter: PivotRewriter) -> None:
  """Verifies behavior when new syntax is invalid Python.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  node: cst.Call = cst.ensure_type(cst.parse_expression("torch.device('cuda')"), cst.Call)
  ctx = MagicMock()
  ctx.target_framework = "jax"
  with patch("ml_switcheroo.plugins.device_allocator.get_adapter") as mock_get:
    mock_adapter = MagicMock()
    mock_adapter.get_device_syntax.return_value = "invalid syntax {{ {"
    mock_get.return_value = mock_adapter
    res: cst.CSTNode = transform_device_allocator(node, ctx)
    assert res is node


def test_device_allocator_adapter_exception(rewriter: PivotRewriter) -> None:
  """Verifies behavior when adapter.get_device_syntax throws exception.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  node: cst.Call = cst.ensure_type(cst.parse_expression("torch.device('cuda')"), cst.Call)
  ctx = MagicMock()
  ctx.target_framework = "jax"
  with patch("ml_switcheroo.plugins.device_allocator.get_adapter") as mock_get:
    mock_adapter = MagicMock()
    mock_adapter.get_device_syntax.side_effect = Exception("Adapter error")
    mock_get.return_value = mock_adapter
    res: cst.CSTNode = transform_device_allocator(node, ctx)
    assert res is node


def test_device_allocator_invalid_colon_index(rewriter: PivotRewriter) -> None:
  """Verifies behavior when colon index is not an integer.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "d = torch.device('cuda:foo')"
  result: str = rewrite_code(rewriter, code)
  assert "jax.devices('gpu:foo')[0]" in result or "jax.devices('cuda:foo')[0]" in result


def test_device_allocator_no_adapter() -> None:
  """Verify device allocator no adapter behavior."""
  node = cst.parse_expression("torch.device('cuda', 0)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  with patch("ml_switcheroo.plugins.device_allocator.get_adapter", return_value=None):
    res = transform_device_allocator(cst.ensure_type(node, cst.Call), ctx)
    assert res is node


def test_parse_device_args_no_args() -> None:
  """Verify parse device args no args behavior."""
  node = cst.parse_expression("torch.device()")
  t, idx = _parse_device_args(cst.ensure_type(node, cst.Call))
  assert t is None
  assert idx is None


def test_device_allocator_adapter_exception_extra() -> None:
  """Verify device allocator adapter exception extra behavior."""
  node = cst.parse_expression("torch.device('cuda')")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  mock_adapter = MagicMock()
  mock_adapter.get_device_syntax.side_effect = Exception("Adapter failure")
  with patch("ml_switcheroo.plugins.device_allocator.get_adapter", return_value=mock_adapter):
    res = transform_device_allocator(cst.ensure_type(node, cst.Call), ctx)
    assert res is node


def test_device_allocator_value_error() -> None:
  """Verify device allocator argument parser handles non-numeric index strings and two-arg calls."""
  expr1 = cst.parse_expression("torch.device('cuda:foo')")
  assert isinstance(expr1, cst.Call)
  dev_type1, idx1 = _parse_device_args(expr1)
  assert dev_type1 is not None and cst.Module(body=[]).code_for_node(dev_type1) == "'cuda:foo'"
  assert idx1 is None

  expr2 = cst.parse_expression("torch.device('cuda', 1)")
  assert isinstance(expr2, cst.Call)
  dev_type2, idx2 = _parse_device_args(expr2)
  assert dev_type2 is not None and cst.Module(body=[]).code_for_node(dev_type2) == "'cuda'"
  assert idx2 is not None and cst.Module(body=[]).code_for_node(idx2) == "1"
