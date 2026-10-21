"""Test suite for the Gather module."""

import typing
from pydantic import JsonValue
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.gather import transform_gather
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Test argument or fixture.
      rewriter: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


@register_framework("custom_fw")
class CustomAdapter:
  """Test helper class CustomAdapter."""

  @property
  def harness_imports(self) -> list[str]:
    """Helper to harness imports.

    Returns:
        Computed test result or fixture value.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Gets harness initialization code.

    Returns:
        Computed test result or fixture value.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Gets to NumPy code.

    Returns:
        Computed test result or fixture value.
    """
    return "return str(obj)"

  @property
  def declared_magic_args(self) -> list[str]:
    """Helper to declared magic arguments.

    Returns:
        Computed test result or fixture value.
    """
    return []


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Execute rewriter factory helper logic.

  Returns:
      Computed test result or fixture value.
  """
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "gather_adapter", getattr({"v": transform_gather}, "__getitem__")("v")
  )
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  gather_def = {
    "variants": {
      "torch": {"api": "torch.gather"},
      "jax": {"api": "jnp.take_along_axis", "requires_plugin": "gather_adapter"},
      "custom_fw": {"api": "custom.gather_nd", "requires_plugin": "gather_adapter"},
    }
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if "gather" in name:
      _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
        {"v": (getattr({"v": "Gather"}, "__getitem__")("v"), getattr({"v": gather_def}, "__getitem__")("v"))},
        "__getitem__",
      )("v")
      return _ret_v
    return None

  mgr.get_definition.side_effect = get_def

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "Gather" and fw in gather_def["variants"]:
      res = gather_def["variants"][fw]
      _ret_v: dict[str, JsonValue] | None = getattr(
        {"v": res if isinstance(getattr({"v": res}, "__getitem__")("v"), dict) else {}}, "__getitem__"
      )("v")
      return _ret_v
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"Gather": gather_def}
  mgr.is_verified.return_value = True
  mgr.framework_configs = {"torch": {}, "jax": {}, "custom_fw": {}}

  def create(target: str) -> PivotRewriter:
    """Creates .

    Args:
        target: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_gather_method_reorder_jax(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of gather method reorder JAX.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "y = x.gather(1, indices)"
  res: str = rewrite_code(rw, code)
  assert "jnp.take_along_axis" in res
  clean: str = res.replace(" ", "")
  assert "(x,indices,1)" in clean or "(x,indices,1,)" in clean


def test_gather_missing_target_passthrough(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of gather missing target passthrough.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rw: PivotRewriter = rewriter_factory("numpy")
  rw.context.hook_context.target_framework = "numpy"
  code: str = "y = torch.gather(x, 1, idx)"
  res: str = rewrite_code(rw, code)
  assert "torch.gather" in res
  assert "jnp" not in res
  assert "take_along_axis" not in res


def test_gather_custom_fw_transpilation(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of gather custom framework transpilation.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rw: PivotRewriter = rewriter_factory("custom_fw")
  code: str = "y = torch.gather(x, 1, idx)"
  res: str = rewrite_code(rw, code)
  assert "custom.gather_nd" in res


# --- Merged from test_gather_missing.py ---


def test_gather_no_target_api() -> None:
  """Verifies the behavior of gather no target API."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None
  node = cst.Call(func=cst.Name("gather"))
  res = transform_gather(node, ctx)
  assert res == node


def test_gather_kwargs() -> None:
  """Verifies the behavior of gather keyword arguments."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.take_along_axis"
  ctx.target_framework = "jax"
  node = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(value=cst.Integer("1"), keyword=cst.Name("dim")),
      cst.Arg(value=cst.Name("idx"), keyword=cst.Name("index")),
    ],
  )
  _res = transform_gather(node, ctx)
  assert isinstance(_res, cst.Call)
  assert cst.Module(body=[]).code_for_node(_res) == "jax.numpy.take_along_axis(x, idx, 1)"


def test_gather_missing_args() -> None:
  """Verifies the behavior of gather missing arguments."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.take_along_axis"
  ctx.target_framework = "jax"
  node = cst.Call(func=cst.Name("gather"), args=[cst.Arg(value=cst.Name("x"))])
  res = transform_gather(node, ctx)
  assert res == node
