"""Test suite for the Gather module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.frameworks.base import StructuralTraits, register_framework
from ml_switcheroo.plugins.gather import transform_gather
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


class CustomAdapter:
  """Test double `CustomAdapter` used in gather tests."""

  @property
  def structural_traits(self) -> StructuralTraits:
    """Return default structural traits for `CustomAdapter`.

    Returns:
        Default `StructuralTraits` instance.
    """
    return StructuralTraits()

  @property
  def harness_imports(self) -> list[str]:
    """Return the `harness_imports` property from `CustomAdapter`.

    Returns:
        Resulting `list[str]` from harness imports.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Gets harness initialization code.

    Returns:
        Resulting `str` from get harness init code.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Gets to NumPy code.

    Returns:
        Resulting `str` from get to numpy code.
    """
    return "return str(obj)"

  @property
  def declared_magic_args(self) -> list[str]:
    """Return the `declared_magic_args` property from `CustomAdapter`.

    Returns:
        Resulting `list[str]` from declared magic args.
    """
    return []


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Create a configured `rewriter_factory` fixture for gather tests.

  Returns:
      Configured `rewriter_factory` fixture value (typing.Callable[[str], PivotRewriter]).
  """
  register_framework("custom_fw")(CustomAdapter)
  hooks.register_hook("gather_adapter")(transform_gather)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  variants_map: dict[str, dict[str, JsonValue]] = {
    "torch": {"api": "torch.gather"},
    "jax": {"api": "jnp.take_along_axis", "requires_plugin": "gather_adapter"},
    "custom_fw": {"api": "custom.gather_nd", "requires_plugin": "gather_adapter"},
  }
  gather_def: dict[str, JsonValue] = {
    "variants": {
      "torch": variants_map["torch"],
      "jax": variants_map["jax"],
      "custom_fw": variants_map["custom_fw"],
    }
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def.
    """
    if "gather" in name:
      return ("Gather", gather_def)
    return None

  mgr.get_definition.side_effect = get_def

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute `resolve` helper in gather tests.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve.
    """
    if aid == "Gather" and fw in variants_map:
      return variants_map[fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"Gather": gather_def}
  mgr.is_verified.return_value = True
  mgr.framework_configs = {"torch": {}, "jax": {}, "custom_fw": {}}

  def create(target: str) -> PivotRewriter:
    """Execute `RuntimeConfig` to support create in gather tests.

    Args:
        target: Target framework or dialect identifier string.

    Returns:
        Resulting `PivotRewriter` from create.
    """
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_gather_method_reorder_jax(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `jnp.take_along_axis` for gather method reorder JAX.

  Args:
      rewriter_factory: Input `rewriter_factory` parameter value (`typing.Callable[[str], PivotRewriter]`).
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "y = x.gather(1, indices)"
  res: str = rewrite_code(rw, code)
  assert "jnp.take_along_axis" in res
  clean: str = res.replace(" ", "")
  assert "(x,indices,1)" in clean or "(x,indices,1,)" in clean


def test_gather_missing_target_passthrough(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` handles gather missing target passthrough as expected.

  Args:
      rewriter_factory: Input `rewriter_factory` parameter value (`typing.Callable[[str], PivotRewriter]`).
  """
  rw: PivotRewriter = rewriter_factory("numpy")
  rw.context.hook_context.target_framework = "numpy"
  code: str = "y = torch.gather(x, 1, idx)"
  res: str = rewrite_code(rw, code)
  assert "torch.gather" in res
  assert "jnp" not in res
  assert "take_along_axis" not in res


def test_gather_custom_fw_transpilation(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `custom.gather_nd` for gather custom fw transpilation.

  Args:
      rewriter_factory: Input `rewriter_factory` parameter value (`typing.Callable[[str], PivotRewriter]`).
  """
  rw: PivotRewriter = rewriter_factory("custom_fw")
  code: str = "y = torch.gather(x, 1, idx)"
  res: str = rewrite_code(rw, code)
  assert "custom.gather_nd" in res


def test_gather_no_target_api() -> None:
  """Verify `Call` handles gather no target API as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None
  node = cst.Call(func=cst.Name("gather"))
  res = transform_gather(node, ctx)
  assert res == node


def test_gather_kwargs() -> None:
  """Verify `Call` handles gather kwargs as expected."""
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
  """Verify `Call` handles gather missing args as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.take_along_axis"
  ctx.target_framework = "jax"
  node = cst.Call(func=cst.Name("gather"), args=[cst.Arg(value=cst.Name("x"))])
  res = transform_gather(node, ctx)
  assert res == node
