"""Module docstring."""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch, PropertyMock
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext, SignatureContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


class DummySemantics(SemanticsManager):
  """Class docstring."""

  framework_configs = {}

  def __init__(self):
    """Function docstring."""
    self.configs = {}
    self.definitions = {}
    self.variants = {}
    self.verified = True
    self._key_origins = {}
    self.framework_configs = self.configs


def get_transformer():
  """Function docstring."""
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True, source_flavour="torch")
  ctx = RewriterContext(semantics, config)
  ctx.hook_context = type(
    "MockHook", (), {"preamble_stmts_mock": [], "inject_preamble": lambda self, s: self.preamble_stmts_mock.append(s)}
  )()
  transformer = ApiTransformer(ctx)
  return transformer, semantics, ctx


def test_api_ctx_property():
  """Function docstring."""
  t, _, ctx = get_transformer()
  assert t.ctx is ctx.hook_context


def test_normalize_arguments_method_call_arg_provided():
  """Function docstring."""
  t, _, _ = get_transformer()
  t._is_module_alias = lambda x: False
  original_node = cst.Call(
    func=cst.Attribute(value=cst.Name("tensor"), attr=cst.Name("method")),
    args=[cst.Arg(keyword=cst.Name("x"), value=cst.Name("a"))],
  )
  op_details = {"std_args": ["input", "other"], "variants": {"torch": {"args": {"input": "x"}}}}

  args = t._normalize_arguments(original_node, original_node, op_details, {})
  assert len(args) == 1
  assert args[0].keyword.value == "input"


def test_normalize_arguments_method_call_no_std_args():
  """Function docstring."""
  t, _, _ = get_transformer()
  t._is_module_alias = lambda x: False
  original_node = cst.Call(func=cst.Attribute(value=cst.Name("tensor"), attr=cst.Name("method")), args=[])
  op_details = {}
  args = t._normalize_arguments(original_node, original_node, op_details, {})
  assert len(args) == 1
  assert isinstance(args[0].value, cst.Name)


def test_normalize_arguments_pack_variadics_no_list_single():
  """Function docstring."""
  t, _, _ = get_transformer()
  t._is_module_alias = lambda x: False
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details = {
    "std_args": [{"name": "dim", "is_variadic": True}],
  }
  api_mapping = {"pack_to_tuple": "dims", "pack_as": "Tuple"}
  args = t._normalize_arguments(original_node, original_node, op_details, api_mapping)
  assert len(args) == 1


def test_normalize_arguments_reconstruct_defaults_error():
  """Function docstring."""
  t, _, _ = get_transformer()
  t._is_module_alias = lambda x: False
  original_node = cst.Call(func=cst.Name("func"), args=[])
  op_details = {
    "std_args": [
      {"name": "input", "default": type("RaiseStr", (), {"__str__": lambda self: (_ for _ in ()).throw(ValueError)})()}
    ],
  }
  args = t._normalize_arguments(original_node, original_node, op_details, {})
  assert len(args) == 0


def test_normalize_arguments_reconstruct_no_alias():
  """Function docstring."""
  t, _, _ = get_transformer()
  t._is_module_alias = lambda x: False
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details = {"std_args": ["input"]}
  api_mapping = {"args": {"input": None}}
  args = t._normalize_arguments(original_node, original_node, op_details, api_mapping)
  assert len(args) == 0


def test_normalize_arguments_reconstruct_val_map():
  """Function docstring."""
  t, _, _ = get_transformer()
  t._is_module_alias = lambda x: False
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(keyword=cst.Name("x"), value=cst.Name("a"))])
  op_details = {"std_args": ["input"], "variants": {"torch": {"args": {"input": "x"}}}}

  api_mapping1 = {"arg_values": {"input": {"a": "b"}}}
  args1 = t._normalize_arguments(original_node, original_node, op_details, api_mapping1)

  api_mapping2 = {"arg_values": {"input": "c + d"}}
  args2 = t._normalize_arguments(original_node, original_node, op_details, api_mapping2)

  api_mapping3 = {"arg_values": {"input": "invalid syntax +++"}}
  args3 = t._normalize_arguments(original_node, original_node, op_details, api_mapping3)

  api_mapping4 = {"arg_values": {"input": 42}}
  args4 = t._normalize_arguments(original_node, original_node, op_details, api_mapping4)


def test_normalize_arguments_reconstruct_different_val():
  """Function docstring."""
  t, _, _ = get_transformer()
  t._is_module_alias = lambda x: False
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details = {"std_args": ["input"]}
  api_mapping = {"arg_values": {"input": 42}}
  args = t._normalize_arguments(original_node, original_node, op_details, api_mapping)
  assert isinstance(args[0].value, cst.Integer)


def test_normalize_arguments_inject_args():
  """Function docstring."""
  t, _, _ = get_transformer()
  t._is_module_alias = lambda x: False
  original_node = cst.Call(func=cst.Name("func"), args=[])
  op_details = {"std_args": []}

  api_mapping = {"arg_values": {"new_arg1": "a + b"}, "inject_args": {"new_arg2": 42, "new_arg3": "invalid_syntax()"}}
  args = t._normalize_arguments(original_node, original_node, op_details, api_mapping)
  assert len(args) == 3


def test_apply_preamble_exception():
  """Function docstring."""
  t, _, _ = get_transformer()
  node = cst.FunctionDef(name=cst.Name("func"), params=cst.Parameters(), body=cst.IndentedBlock(body=[]))
  res = t._apply_preamble(node, ["invalid syntax +++"])
  assert len(res.body.body) == 0


def test_inject_stmts_to_body():
  """Function docstring."""
  t, _, _ = get_transformer()
  node = cst.FunctionDef(name=cst.Name("func"), params=cst.Parameters(), body=cst.SimpleStatementSuite(body=[cst.Pass()]))
  res = t._inject_stmts_to_body(node, [cst.SimpleStatementLine(body=[cst.Expr(cst.Integer("1"))])])
  assert isinstance(res.body, cst.IndentedBlock)

  docstring_stmt = cst.SimpleStatementLine(body=[cst.Expr(cst.SimpleString('"""doc"""'))])
  node2 = cst.FunctionDef(
    name=cst.Name("func"),
    params=cst.Parameters(),
    body=cst.IndentedBlock(body=[docstring_stmt, cst.SimpleStatementLine(body=[cst.Pass()])]),
  )
  res2 = t._inject_stmts_to_body(node2, [cst.SimpleStatementLine(body=[cst.Expr(cst.Integer("1"))])])
  assert len(res2.body.body) == 3
  assert isinstance(res2.body.body[0].body[0].value, cst.SimpleString)
