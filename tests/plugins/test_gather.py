"""Test suite for the Gather module."""

import typing
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
  """Rewrites code."""
  return typing.cast(str, rewriter.convert(cst.parse_module(code)).code)


@register_framework("custom_fw")
class CustomAdapter:
  """Docstring."""

  @property
  def harness_imports(self) -> list[str]:
    """Helper to harness imports."""
    return []

  def get_harness_init_code(self) -> str:
    """Gets harness initialization code."""
    return ""

  def get_to_numpy_code(self) -> str:
    """Gets to NumPy code."""
    return "return str(obj)"

  @property
  def declared_magic_args(self) -> list[str]:
    """Helper to declared magic arguments."""
    return []


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Docstring."""
  hooks._HOOKS["gather_adapter"] = transform_gather
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  gather_def: dict[str, typing.Any] = {
    "variants": {
      "torch": {"api": "torch.gather"},
      "jax": {"api": "jnp.take_along_axis", "requires_plugin": "gather_adapter"},
      "custom_fw": {"api": "custom.gather_nd", "requires_plugin": "gather_adapter"},
    }
  }

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def."""
    if "gather" in name:
      return ("Gather", gather_def)
    return None

  mgr.get_definition.side_effect = get_def

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves ."""
    if aid == "Gather" and fw in gather_def["variants"]:
      return typing.cast(dict[str, typing.Any], gather_def["variants"][fw])
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"Gather": gather_def}
  mgr.is_verified.return_value = True
  mgr.framework_configs = {"torch": {}, "jax": {}, "custom_fw": {}}

  def create(target: str) -> PivotRewriter:
    """Creates ."""
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_gather_method_reorder_jax(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of gather method reorder JAX."""
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "y = x.gather(1, indices)"
  res: str = rewrite_code(rw, code)
  assert "jnp.take_along_axis" in res
  clean: str = res.replace(" ", "")
  assert "(x,indices,1)" in clean or "(x,indices,1,)" in clean


def test_gather_missing_target_passthrough(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of gather missing target passthrough."""
  rw: PivotRewriter = rewriter_factory("numpy")
  rw.context.hook_context.target_fw = "numpy"
  code: str = "y = torch.gather(x, 1, idx)"
  res: str = rewrite_code(rw, code)
  assert "torch.gather" in res
  assert "jnp" not in res
  assert "take_along_axis" not in res


def test_gather_custom_fw_transpilation(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of gather custom framework transpilation."""
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
  res: typing.Any = transform_gather(node, ctx)
  assert res == node


def test_gather_kwargs() -> None:
  """Verifies the behavior of gather keyword arguments."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.take_along_axis"
  ctx.target_fw = "jax"
  node = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(value=cst.Integer("1"), keyword=cst.Name("dim")),
      cst.Arg(value=cst.Name("idx"), keyword=cst.Name("index")),
    ],
  )
  transform_gather(node, ctx)


def test_gather_missing_args() -> None:
  """Verifies the behavior of gather missing arguments."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.take_along_axis"
  ctx.target_fw = "jax"
  node = cst.Call(func=cst.Name("gather"), args=[cst.Arg(value=cst.Name("x"))])
  res: typing.Any = transform_gather(node, ctx)
  assert res == node
