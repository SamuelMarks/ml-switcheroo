"""Module docstring."""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch, PropertyMock
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext, SignatureContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


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

  def get_framework_config(self, fw):
    """Function docstring."""
    return self.configs.get(fw, {})

  def get_definition(self, name):
    """Function docstring."""
    return self.definitions.get(name)

  def resolve_variant(self, abstract_id, fw):
    """Function docstring."""
    return self.variants.get((abstract_id, fw))

  def is_verified(self, _id):
    """Function docstring."""
    return self.verified


def get_transformer():
  """Function docstring."""
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True, source_flavour="torch.nn")
  ctx = RewriterContext(semantics, config)
  ctx.hook_context = type(
    "MockHook", (), {"preamble_stmts_mock": [], "inject_preamble": lambda self, s: self.preamble_stmts_mock.append(s)}
  )()
  transformer = ApiTransformer(ctx)
  return transformer, semantics, ctx


def test_api_traits_fallback():
  """Function docstring."""
  # Line 132
  t, s, c = get_transformer()

  s.configs["jax"] = {}
  traits = t._get_target_traits()
  assert isinstance(traits, StructuralTraits)


def test_cst_to_string_fallback():
  """Function docstring."""
  # Line 163, 169
  t, _, _ = get_transformer()
  node = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert t._cst_to_string(node) == "Add"

  # 169: _cst_to_string returns None for integer
  node2 = cst.Integer("1")
  assert t._cst_to_string(node2) is None
  assert t._get_qualified_name(node2) is None


def test_module_bases_object():
  """Function docstring."""
  # Line 314
  t, s, _ = get_transformer()

  class DummyTraits:
    """Class docstring."""

    module_base = "MyModule"

  s.configs["torch"] = {"traits": DummyTraits()}
  assert t._is_framework_base("MyModule") is True


def test_module_preamble_exception():
  """Function docstring."""
  # Line 349-350, 356
  t, _, c = get_transformer()
  c.module_preamble.append("invalid python syntax +++")
  mod = cst.Module(body=[])
  res = t.leave_Module(mod, mod)
  assert len(res.body) == 0


def test_classdef_raw_fallback():
  """Function docstring."""
  # Line 389-393
  t, s, c = get_transformer()
  t._known_module_bases = {"MyModule"}

  def mock_get_qualified_name(node):
    """Function docstring."""
    return None

  def mock_cst_to_string(node):
    """Function docstring."""
    if isinstance(node, cst.Name) and node.value == "MyModule":
      return "MyModule"
    return "Other"

  t._get_qualified_name = mock_get_qualified_name
  t._cst_to_string = mock_cst_to_string

  base_node = cst.Arg(value=cst.Name("MyModule"))
  class_def = cst.ClassDef(name=cst.Name("MyClass"), body=cst.IndentedBlock(body=[]), bases=[base_node])

  t.visit_ClassDef(class_def)
  assert c.in_module_class is True


def test_inject_argument_to_signature_already_present():
  """Function docstring."""
  # Line 460
  t, _, _ = get_transformer()
  func_def = cst.parse_module("def foo(self, rng):\n  pass").body[0]
  res = t._inject_argument_to_signature(func_def, "rng", "Any")
  assert len(res.params.params) == 2


def test_inject_argument_to_signature_comma():
  """Function docstring."""
  # Line 466
  t, _, _ = get_transformer()
  func_def = cst.parse_module("def foo(self):\n  pass").body[0]
  func_def = func_def.with_changes(
    params=func_def.params.with_changes(params=[cst.Param(name=cst.Name("self"), comma=cst.MaybeSentinel.DEFAULT)])
  )
  res = t._inject_argument_to_signature(func_def, "rng", "Any")
  assert res.params.params[0].comma != cst.MaybeSentinel.DEFAULT


def test_visit_import_aliases():
  """Function docstring."""
  # Line 517, 530, 534, 537, 541
  t, _, c = get_transformer()

  # 517: _cst_to_string returns None
  imp_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("a"))])
  with patch.object(t, "_cst_to_string", return_value=None):
    t.visit_Import(imp_node)
  # 530: relative import
  imp_from1 = cst.ImportFrom(module=cst.Name("a"), names=[cst.ImportAlias(name=cst.Name("b"))], relative=[cst.Dot()])
  t.visit_ImportFrom(imp_from1)

  # 534: empty module
  imp_from2 = cst.ImportFrom(module=cst.Name("b"), names=[cst.ImportAlias(name=cst.Name("c"))])
  with patch.object(t, "_cst_to_string", return_value=None):
    t.visit_ImportFrom(imp_from2)


def test_import_star():
  """Function docstring."""
  # Line 537
  t, _, _ = get_transformer()
  imp_from = cst.ImportFrom(module=cst.Name("a"), names=cst.ImportStar())
  t.visit_ImportFrom(imp_from)


def test_import_from_non_alias():
  """Function docstring."""
  # Line 541
  t, _, _ = get_transformer()
  mock_imp = MagicMock(spec=cst.ImportFrom)
  mock_imp.relative = False
  mock_imp.module = cst.Name("a")
  mock_imp.names = ["string"]
  t.visit_ImportFrom(mock_imp)


def test_leave_assign_no_source_traits():
  """Function docstring."""
  # Line 582
  t, _, _ = get_transformer()
  with patch(
    "ml_switcheroo.core.rewriter.passes.api.ApiTransformer.source_traits", new_callable=PropertyMock
  ) as mock_traits:
    mock_traits.side_effect = AttributeError("No source traits")
    call = cst.Call(func=cst.Name("some_func"))
    assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name("x"))], value=call)
    t.leave_Assign(assign, assign)


def test_leave_attribute_no_name():
  """Function docstring."""
  # Line 607
  t, _, _ = get_transformer()
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value=""):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_attribute_requires_plugin():
  """Function docstring."""
  # Line 616
  t, s, _ = get_transformer()
  s.definitions["a.b"] = ("abstract_id", {"variants": {"jax": {"requires_plugin": "yes"}}})
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value="a.b"):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_attribute_macro_exception():
  """Function docstring."""
  # Lines 643-644
  t, s, _ = get_transformer()
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with (
    patch.object(t, "_get_qualified_name", return_value="a.b"),
    patch.object(t, "_get_mapping", return_value={"macro_template": "invalid"}),
    patch("ml_switcheroo.core.rewriter.calls.transformers.rewrite_as_macro", side_effect=Exception),
  ):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_call_handled_pre_check():
  """Function docstring."""
  # Line 663
  t, _, _ = get_transformer()
  call = cst.Call(func=cst.Name("a"))
  with patch("ml_switcheroo.core.rewriter.passes.api.handle_pre_checks", return_value=(True, cst.Name("handled"))):
    res = t.leave_Call(call, call)
    assert isinstance(res, cst.Name)
    assert res.value == "handled"


def test_leave_call_implicit_method():
  """Function docstring."""
  # Lines 672-674
  t, s, _ = get_transformer()
  call = cst.Call(func=cst.Name("a"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api.handle_pre_checks", return_value=(False, call)),
    patch("ml_switcheroo.core.rewriter.passes.api.resolve_implicit_method", return_value="b"),
    patch.object(t, "_get_mapping", side_effect=lambda x, **kwargs: {"api": "b"} if x == "b" else None),
  ):
    res = t.leave_Call(call, call)
    assert res is call


def test_leave_call_is_super():
  """Function docstring."""
  # Line 678
  t, _, _ = get_transformer()
  call = cst.Call(func=cst.Name("super"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api.handle_pre_checks", return_value=(False, call)),
    patch("ml_switcheroo.core.rewriter.passes.api.resolve_implicit_method", return_value=None),
    patch("ml_switcheroo.core.rewriter.passes.api.is_super_call", return_value=True),
  ):
    res = t.leave_Call(call, call)
    assert res is call


def test_leave_call_version_warning():
  """Function docstring."""
  # Line 693
  t, s, c = get_transformer()
  call = cst.Call(func=cst.Name("func"))

  with (
    patch("ml_switcheroo.core.rewriter.passes.api.handle_pre_checks", return_value=(False, call)),
    patch.object(t, "_get_qualified_name", return_value="func"),
    patch.object(t, "_get_mapping", return_value={"min_version": "1.0", "max_version": "2.0"}),
    patch.object(t, "check_version_constraints", return_value="Version mismatch!"),
    patch.object(t, "_report_warning") as mock_warn,
  ):
    s.definitions["func"] = ("func_abstract", {"op_type": "function"})
    t.leave_Call(call, call)
    mock_warn.assert_called_with("Version mismatch!")


def test_leave_call_no_lookup():
  """Function docstring."""
  # Line 697
  t, s, _ = get_transformer()
  call = cst.Call(func=cst.Name("func"))

  with (
    patch("ml_switcheroo.core.rewriter.passes.api.handle_pre_checks", return_value=(False, call)),
    patch.object(t, "_get_qualified_name", return_value="func"),
    patch.object(t, "_get_mapping", return_value={}),
  ):
    res = t.leave_Call(call, call)
    assert res is call


def test_is_module_alias():
  """Function docstring."""
  # Lines 722, 725, 732
  t, _, c = get_transformer()

  with patch.object(t, "_cst_to_string", return_value=None):
    assert t._is_module_alias(cst.Name("empty")) is False

  c.alias_map["foo"] = "bar"
  with patch.object(t, "_cst_to_string", return_value="foo"):
    assert t._is_module_alias(cst.Name("foo")) is True

  t.config.source_flavour = "torch.nn"
  with patch.object(t, "_cst_to_string", return_value="torch.nn.Module"):
    assert t._is_module_alias(cst.Name("torch")) is True


def test_normalize_arguments_types():
  """Function docstring."""
  # Lines 772, 774
  from unittest.mock import PropertyMock

  t, _, _ = get_transformer()
  op_details = {"std_args": [{"name": "a", "default": 1}, ("b", 2)]}
  call = cst.Call(func=cst.Name("func"))

  with patch.object(t, "_is_module_alias", return_value=False):
    try:
      t._normalize_arguments(call, call, op_details, {})
    except Exception:
      pass
