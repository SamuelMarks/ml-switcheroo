"""Test suite for the Api Extra2 module."""

from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments
import libcst as cst
from unittest.mock import MagicMock, patch, PropertyMock
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


class DummySemantics(SemanticsManager):
  """Dummy Semantics class for testing purposes."""

  framework_configs = {}

  def __init__(self):
    """Initializes the DummySemantics instance."""
    self.configs = {}
    self.definitions = {}
    self.variants = {}
    self.verified = True
    self._key_origins = {}
    self.framework_configs = self.configs

  def get_framework_config(self, fw):
    """Mock implementation of get framework configuration."""
    return self.configs.get(fw, {})

  def get_definition(self, name):
    """Mock implementation of get definition."""
    return self.definitions.get(name)

  def resolve_variant(self, abstract_id, fw):
    """Mock implementation of resolve variant."""
    return self.variants.get((abstract_id, fw))

  def is_verified(self, _id):
    """Mock implementation of is verified."""
    return self.verified


def get_transformer():
  """Gets transformer."""
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True, source_flavour="torch.nn")
  ctx = RewriterContext(semantics, config)
  ctx.hook_context = type(
    "MockHook", (), {"preamble_stmts_mock": [], "inject_preamble": lambda self, s: self.preamble_stmts_mock.append(s)}
  )()
  transformer = ApiTransformer(ctx)
  return (transformer, semantics, ctx)


def test_api_traits_fallback():
  """Verifies the behavior of API traits fallback."""
  (t, s, c) = get_transformer()
  s.configs["jax"] = {}
  traits = t._get_target_traits()
  assert isinstance(traits, StructuralTraits)


def test_cst_to_string_fallback():
  """Verifies the behavior of cst to string fallback."""
  (t, _, _) = get_transformer()
  node = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert t._cst_to_string(node) == "Add"
  node2 = cst.Integer("1")
  assert t._cst_to_string(node2) is None
  assert t._get_qualified_name(node2) is None


def test_module_bases_object():
  """Verifies the behavior of module bases object."""
  (t, s, _) = get_transformer()

  class DummyTraits:
    """Dummy Traits class for testing purposes."""

    module_base = "MyModule"

  s.configs["torch"] = {"traits": DummyTraits()}
  assert t._is_framework_base("MyModule") is True


def test_module_preamble_exception():
  """Verifies the behavior of module preamble correctly handling an exception."""
  (t, _, c) = get_transformer()
  c.module_preamble.append("invalid python syntax +++")
  mod = cst.Module(body=[])
  res = t.leave_Module(mod, mod)
  assert len(res.body) == 0


def test_classdef_raw_fallback():
  """Verifies the behavior of classdef raw fallback."""
  (t, s, c) = get_transformer()
  t._known_module_bases = {"MyModule"}

  def mock_get_qualified_name(node):
    """Provides a mock get qualified name for testing."""
    return None

  def mock_cst_to_string(node):
    """Provides a mock cst to string for testing."""
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
  """Injects argument to signature already present."""
  (t, _, _) = get_transformer()
  func_def = cst.parse_module("def foo(self, rng):\n  pass").body[0]
  res = t._inject_argument_to_signature(func_def, "rng", "Any")
  assert len(res.params.params) == 2


def test_inject_argument_to_signature_comma():
  """Injects argument to signature comma."""
  (t, _, _) = get_transformer()
  func_def = cst.parse_module("def foo(self):\n  pass").body[0]
  func_def = func_def.with_changes(
    params=func_def.params.with_changes(params=[cst.Param(name=cst.Name("self"), comma=cst.MaybeSentinel.DEFAULT)])
  )
  res = t._inject_argument_to_signature(func_def, "rng", "Any")
  assert res.params.params[0].comma != cst.MaybeSentinel.DEFAULT


def test_visit_import_aliases():
  """Verifies the behavior of visit import aliases."""
  (t, _, c) = get_transformer()
  imp_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("a"))])
  with patch.object(t, "_cst_to_string", return_value=None):
    t.visit_Import(imp_node)
  imp_from1 = cst.ImportFrom(module=cst.Name("a"), names=[cst.ImportAlias(name=cst.Name("b"))], relative=[cst.Dot()])
  t.visit_ImportFrom(imp_from1)
  imp_from2 = cst.ImportFrom(module=cst.Name("b"), names=[cst.ImportAlias(name=cst.Name("c"))])
  with patch.object(t, "_cst_to_string", return_value=None):
    t.visit_ImportFrom(imp_from2)


def test_import_star():
  """Verifies the behavior of import star."""
  (t, _, _) = get_transformer()
  imp_from = cst.ImportFrom(module=cst.Name("a"), names=cst.ImportStar())
  t.visit_ImportFrom(imp_from)


def test_import_from_non_alias():
  """Verifies the behavior of import from non alias."""
  (t, _, _) = get_transformer()
  mock_imp = MagicMock(spec=cst.ImportFrom)
  mock_imp.relative = False
  mock_imp.module = cst.Name("a")
  mock_imp.names = ["string"]
  t.visit_ImportFrom(mock_imp)


def test_leave_assign_no_source_traits():
  """Verifies the behavior of leave assign no source traits."""
  (t, _, _) = get_transformer()
  with patch(
    "ml_switcheroo.core.rewriter.passes.api.ApiTransformer.source_traits", new_callable=PropertyMock
  ) as mock_traits:
    mock_traits.side_effect = AttributeError("No source traits")
    call = cst.Call(func=cst.Name("some_func"))
    assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name("x"))], value=call)
    t.leave_Assign(assign, assign)


def test_leave_attribute_no_name():
  """Verifies the behavior of leave attribute no name."""
  (t, _, _) = get_transformer()
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value=""):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_attribute_requires_plugin():
  """Verifies the behavior of leave attribute requires plugin."""
  (t, s, _) = get_transformer()
  s.definitions["a.b"] = ("abstract_id", {"variants": {"jax": {"requires_plugin": "yes"}}})
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value="a.b"):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_attribute_macro_exception():
  """Verifies the behavior of leave attribute macro correctly handling an exception."""
  (t, s, _) = get_transformer()
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with (
    patch.object(t, "_get_qualified_name", return_value="a.b"),
    patch.object(t, "_get_mapping", return_value={"macro_template": "invalid"}),
    patch("ml_switcheroo.core.rewriter.calls.transformers.rewrite_as_macro", side_effect=Exception),
  ):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_call_handled_pre_check():
  """Verifies the behavior of leave call handled pre check."""
  (t, _, _) = get_transformer()
  call = cst.Call(func=cst.Name("a"))
  with patch(
    "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(True, cst.Name("handled"))
  ):
    res = t.leave_Call(call, call)
    assert isinstance(res, cst.Name)
    assert res.value == "handled"


def test_leave_call_implicit_method():
  """Verifies the behavior of leave call implicit method."""
  (t, s, _) = get_transformer()
  call = cst.Call(func=cst.Name("a"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method", return_value="b"),
    patch.object(t, "_get_mapping", side_effect=lambda x, **kwargs: {"api": "b"} if x == "b" else None),
  ):
    res = t.leave_Call(call, call)
    assert res is call


def test_leave_call_is_super():
  """Verifies the behavior of leave call is super."""
  (t, _, _) = get_transformer()
  call = cst.Call(func=cst.Name("super"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method", return_value=None),
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.is_super_call", return_value=True),
  ):
    res = t.leave_Call(call, call)
    assert res is call


def test_leave_call_version_warning():
  """Verifies the behavior of leave call version warning."""
  (t, s, c) = get_transformer()
  call = cst.Call(func=cst.Name("func"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch.object(t, "_get_qualified_name", return_value="func"),
    patch.object(t, "_get_mapping", return_value={"min_version": "1.0", "max_version": "2.0"}),
    patch.object(t, "check_version_constraints", return_value="Version mismatch!"),
    patch.object(t, "_report_warning") as mock_warn,
  ):
    s.definitions["func"] = ("func_abstract", {"op_type": "function"})
    t.leave_Call(call, call)
    mock_warn.assert_called_with("Version mismatch!")


def test_leave_call_no_lookup():
  """Verifies the behavior of leave call no lookup."""
  (t, s, _) = get_transformer()
  call = cst.Call(func=cst.Name("func"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch.object(t, "_get_qualified_name", return_value="func"),
    patch.object(t, "_get_mapping", return_value={}),
  ):
    res = t.leave_Call(call, call)
    assert res is call


def test_is_module_alias():
  """Checks if is module alias."""
  (t, _, c) = get_transformer()
  with patch.object(t, "_cst_to_string", return_value=None):
    assert t._is_module_alias(cst.Name("empty")) is False
  c.alias_map["foo"] = "bar"
  with patch.object(t, "_cst_to_string", return_value="foo"):
    assert t._is_module_alias(cst.Name("foo")) is True
  t.config.source_flavour = "torch.nn"
  with patch.object(t, "_cst_to_string", return_value="torch.nn.Module"):
    assert t._is_module_alias(cst.Name("torch")) is True


def test_normalize_arguments_types():
  """Verifies the behavior of normalize arguments types."""
  (t, _, _) = get_transformer()
  op_details = {"std_args": [{"name": "a", "default": 1}, ("b", 2)]}
  call = cst.Call(func=cst.Name("func"))
  with patch.object(t, "_is_module_alias", return_value=False):
    try:
      normalize_arguments(call, call, op_details, {})
    except Exception:
      pass
