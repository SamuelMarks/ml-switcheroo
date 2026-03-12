import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.reshape import _create_dotted_name, transform_view_semantics


def test_create_dotted_name():
  node = _create_dotted_name("np.reshape")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "reshape"
  assert isinstance(node.value, cst.Name)
  assert node.value.value == "np"

  node_single = _create_dotted_name("reshape")
  assert isinstance(node_single, cst.Name)
  assert node_single.value == "reshape"


def test_transform_view_semantics_no_mapping():
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  node = cst.Call(func=cst.Name("view"), args=[])
  result = transform_view_semantics(node, ctx)
  assert result is node


def test_transform_view_semantics_method_empty_args():
  ctx = MagicMock()
  ctx.lookup_api.side_effect = lambda x: "jnp.reshape" if x == "Reshape" else None
  ctx._runtime_config.strict_mode = False

  # x.view()
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")), args=[])
  result = transform_view_semantics(node, ctx)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "reshape"
  assert len(result.args) == 1
  assert result.args[0].value.value == "x"


def test_transform_view_semantics_method_pack_varargs():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"
  ctx._runtime_config.strict_mode = False

  # x.view(1, 2)
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")),
    args=[cst.Arg(value=cst.Integer("1")), cst.Arg(value=cst.Integer("2"))],
  )
  result = transform_view_semantics(node, ctx)
  assert len(result.args) == 2
  assert result.args[0].value.value == "x"
  assert isinstance(result.args[1].value, cst.Tuple)
  assert len(result.args[1].value.elements) == 2


def test_transform_view_semantics_method_pack_single_int():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"
  ctx._runtime_config.strict_mode = False

  # x.view(1)
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")), args=[cst.Arg(value=cst.Integer("1"))])
  result = transform_view_semantics(node, ctx)
  assert len(result.args) == 2
  assert isinstance(result.args[1].value, cst.Tuple)
  assert len(result.args[1].value.elements) == 1
  assert result.args[1].value.elements[0].value.value == "1"


def test_transform_view_semantics_method_no_pack_tuple():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"
  ctx._runtime_config.strict_mode = False

  # x.view((1, 2))
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")),
    args=[cst.Arg(value=cst.Tuple(elements=[cst.Element(value=cst.Integer("1")), cst.Element(value=cst.Integer("2"))]))],
  )
  result = transform_view_semantics(node, ctx)
  assert len(result.args) == 2
  assert result.args[0].value.value == "x"
  assert isinstance(result.args[1].value, cst.Tuple)


def test_transform_view_semantics_func_empty_args():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"

  # view()
  node = cst.Call(func=cst.Name("view"), args=[])
  result = transform_view_semantics(node, ctx)
  # Returns node directly if no args
  assert result is node


def test_transform_view_semantics_func_no_pack():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"
  ctx._runtime_config.strict_mode = False

  # view(x, (1, 2))
  node = cst.Call(
    func=cst.Name("view"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(value=cst.Tuple(elements=[cst.Element(value=cst.Integer("1")), cst.Element(value=cst.Integer("2"))])),
    ],
  )
  result = transform_view_semantics(node, ctx)
  assert len(result.args) == 2
  assert result.args[0].value.value == "x"
  assert isinstance(result.args[1].value, cst.Tuple)


def test_transform_view_semantics_func_pack_varargs():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"
  ctx._runtime_config.strict_mode = False

  # view(x, 1, 2)
  node = cst.Call(
    func=cst.Name("view"),
    args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Integer("1")), cst.Arg(value=cst.Integer("2"))],
  )
  result = transform_view_semantics(node, ctx)
  assert len(result.args) == 2
  assert result.args[0].value.value == "x"
  assert isinstance(result.args[1].value, cst.Tuple)
  assert len(result.args[1].value.elements) == 2


def test_transform_view_semantics_func_pack_single_int():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"
  ctx._runtime_config.strict_mode = False

  # view(x, 1)
  node = cst.Call(func=cst.Name("view"), args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Integer("1"))])
  result = transform_view_semantics(node, ctx)
  assert len(result.args) == 2
  assert result.args[0].value.value == "x"
  assert isinstance(result.args[1].value, cst.Tuple)
  assert len(result.args[1].value.elements) == 1


def test_transform_view_semantics_func_empty_orig_args():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"
  ctx._runtime_config.strict_mode = False

  # view(x)
  node = cst.Call(func=cst.Name("view"), args=[cst.Arg(value=cst.Name("x"))])
  result = transform_view_semantics(node, ctx)
  assert len(result.args) == 1
  assert result.args[0].value.value == "x"


def test_transform_view_semantics_strict_mode():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"
  ctx._runtime_config.strict_mode = True
  ctx.plugin_traits.strict_materialization_method = "block_until_ready"

  # x.view(1)
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")), args=[cst.Arg(value=cst.Integer("1"))])
  result = transform_view_semantics(node, ctx)
  # Output should be jnp.reshape(x, (1,)).block_until_ready()
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "block_until_ready"
  assert len(result.args) == 0
  assert isinstance(result.func.value, cst.Call)
  inner_call = result.func.value
  assert isinstance(inner_call.func, cst.Attribute)
  assert inner_call.func.attr.value == "reshape"
  assert len(inner_call.args) == 2


def test_transform_view_semantics_strict_mode_no_trait():
  ctx = MagicMock()
  ctx.lookup_api.return_value = "jnp.reshape"
  ctx._runtime_config.strict_mode = True
  ctx.plugin_traits.strict_materialization_method = None

  # x.view(1)
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")), args=[cst.Arg(value=cst.Integer("1"))])
  result = transform_view_semantics(node, ctx)
  # Output should NOT have block_until_ready
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "reshape"
  assert len(result.args) == 2
