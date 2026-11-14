"""Test suite for the Keras Sequential plugin."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.keras_sequential import transform_keras_sequential


def _make_ctx(api: str | None) -> HookContext:
  """Construct a typed ``HookContext`` with an optional ``Sequential`` variant API.

  Args:
      api: Optional dotted Sequential API string to expose via semantics.

  Returns:
      Configured ``HookContext`` instance.
  """
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": api} if api else None
  ctx = HookContext(semantics=semantics, config=RuntimeConfig(source_framework="torch", target_framework="keras"))
  ctx.current_op_id = "Sequential"
  return ctx


def test_keras_sequential_missing_api_aborts_safely() -> None:
  """Verify unmapped Sequential calls return unchanged without hardcoded fallback."""
  node = cst.Call(func=cst.Name("Sequential"), args=[cst.Arg(cst.Name("L"))])
  ctx = _make_ctx(None)
  res = transform_keras_sequential(node, ctx)
  assert res is node


def test_keras_sequential_empty_args() -> None:
  """Verify Sequential() with zero args rewrites the target class only."""
  node = cst.Call(func=cst.Name("Sequential"), args=[])
  res = transform_keras_sequential(node, _make_ctx("my.Seq"))
  assert isinstance(res.func, cst.Attribute)
  assert isinstance(res.func.value, cst.Name)
  assert res.func.value.value == "my"
  assert res.func.attr.value == "Seq"
  assert not res.args


def test_keras_sequential_list_and_keyword_sequence_args() -> None:
  """Verify pre-packed list args and keyword layer sequence args are preserved."""
  node = cst.Call(func=cst.Name("Sequential"), args=[cst.Arg(cst.List([cst.Element(cst.Name("L"))]))])
  res = transform_keras_sequential(node, _make_ctx("my.Seq"))
  assert isinstance(res.args[0].value, cst.List)

  kw_seq = cst.Call(
    func=cst.Name("Sequential"),
    args=[cst.Arg(keyword=cst.Name("layers"), value=cst.List([cst.Element(cst.Name("L"))]))],
  )
  res_kw = transform_keras_sequential(kw_seq, _make_ctx("my.Seq"))
  assert len(res_kw.args) == 1
  assert res_kw.args[0].keyword is None
  assert isinstance(res_kw.args[0].value, cst.List)


def test_keras_sequential_keyword_only_non_layer_args() -> None:
  """Verify non-layer keyword arguments without positional layers emit empty args."""
  ctx = HookContext(semantics=None, config=RuntimeConfig(source_framework="torch", target_framework="keras"))
  mock_ctx = MagicMock(spec=HookContext)
  mock_ctx.semantics = None
  mock_ctx.current_op_id = None
  mock_ctx.lookup_api.return_value = None
  mock_ctx.current_variant = FrameworkVariant(api="my.Seq")

  only_name = cst.Call(
    func=cst.Name("Sequential"),
    args=[cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString("'m'"))],
  )
  res_only_name = transform_keras_sequential(only_name, mock_ctx)
  assert len(res_only_name.args) == 0
  assert transform_keras_sequential(only_name, ctx) is only_name

  mixed = cst.Call(
    func=cst.Name("Sequential"),
    args=[cst.Arg(value=cst.Name("L1")), cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString("'test'"))],
  )
  res_mixed = transform_keras_sequential(mixed, _make_ctx("my.Seq"))
  assert isinstance(res_mixed.args[0].value, cst.List)
  assert len(res_mixed.args[0].value.elements) == 1
