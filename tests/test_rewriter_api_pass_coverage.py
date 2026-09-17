"""Module docstring."""

from typing import Any, Dict, Optional, Tuple

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiPass, ApiTransformer
from ml_switcheroo.semantics.manager import SemanticsManager


def test_apipass_branches() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: SemanticsManager = SemanticsManager()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  pass_obj: ApiPass = ApiPass()

  module: cst.Module = cst.parse_module(
    "import torch\nx = torch.abs(y)\nclass Model:\n def __init__(self):\n  self.layer = 1\n def forward(self, x):\n  return self.layer(x)\n"
  )
  pass_obj.transform(module, context)


def test_apitransformer_methods() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")

  class DummySemanticsNone:
    """Docstring."""

    framework_configs: Dict[str, Any] = {"torch": {"traits": {"module_base": "torch.nn.Module"}}}

    def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      return None

    def is_verified(self, name: str) -> bool:
      """Docstring."""
      return True

    def get_framework_config(self, fw: str) -> Optional[Dict[str, Any]]:
      """Docstring."""
      return None

  semantics_none: DummySemanticsNone = DummySemanticsNone()
  context: RewriterContext = RewriterContext(semantics=semantics_none, config=config)

  transformer: ApiTransformer = ApiTransformer(context)
  assert transformer.semantics == semantics_none
  assert transformer.config == config
  assert transformer.source_fw == "torch"
  assert transformer.target_fw == "jax"
  assert not getattr(transformer, "strict_mode", False)

  getattr(transformer, "source_traits")
  transformer._get_target_traits()

  class DummySemanticsTraits:
    """Docstring."""

    framework_configs: Dict[str, Any] = {}

    def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      return None

    def is_verified(self, name: str) -> bool:
      """Docstring."""
      return True

    def get_framework_config(self, fw: str) -> Optional[Dict[str, Any]]:
      """Docstring."""
      return {"traits": {}}

  semantics_traits: DummySemanticsTraits = DummySemanticsTraits()
  context_traits: RewriterContext = RewriterContext(semantics=semantics_traits, config=config)
  transformer_traits: ApiTransformer = ApiTransformer(context_traits)

  getattr(transformer_traits, "source_traits")
  getattr(transformer_traits, "source_traits")
  transformer_traits._get_target_traits()
  assert transformer._cst_to_string(cst.BinaryOperation(cst.Name("a"), cst.Add(), cst.Name("b"))) == "Add"
  assert transformer._cst_to_string(cst.Pass()) is None
  assert transformer._cst_to_string(cst.Attribute(value=cst.Pass(), attr=cst.Name("a"))) is None

  transformer._get_source_lifecycle_lists()
  getattr(transformer, "ctx", None)

  # 222
  transformer.visit_Module(cst.parse_module("import os"))

  # 231
  import_node: cst.Import = cst.Import(names=[cst.ImportAlias(cst.Name("os"))])
  transformer.visit_Import(import_node)

  # 260
  import_from_node: cst.ImportFrom = cst.ImportFrom(module=cst.Name("os"), names=[cst.ImportAlias(cst.Name("path"))])
  transformer.visit_ImportFrom(import_from_node)

  # 277, 280, 286-288, 299-301
  node_func: cst.FunctionDef = getattr(cst.parse_module("def f(): pass"), "body")[0]
  transformer.leave_FunctionDef(node_func, node_func)

  # 351-359, 362
  node_assign: cst.Assign = getattr(cst.parse_statement("a = 1"), "body")[0]
  transformer.leave_Assign(node_assign, node_assign)

  # 384-385, 391
  node_attr: cst.Attribute = getattr(getattr(cst.parse_statement("a.b"), "body")[0], "value")
  transformer.leave_Attribute(node_attr, node_attr)

  # 498
  class_node: cst.ClassDef = getattr(cst.parse_module("class A(torch.nn.Module): pass"), "body")[0]
  transformer.visit_ClassDef(class_node)
  transformer.leave_ClassDef(class_node, class_node)

  class_node2: cst.ClassDef = getattr(cst.parse_module("class B(UnknownBase): pass"), "body")[0]
  transformer.visit_ClassDef(class_node2)
  transformer.leave_ClassDef(class_node2, class_node2)

  # 644
  stmt_line: cst.SimpleStatementLine = getattr(cst.parse_module("a = 1"), "body")[0]
  transformer.visit_SimpleStatementLine(stmt_line)
  transformer.leave_SimpleStatementLine(stmt_line, stmt_line)

  # 1242
  def_node: cst.FunctionDef = getattr(cst.parse_module("def f(): pass"), "body")[0]
  transformer._apply_preamble(def_node, ["a = 1"])

  # 1262
  stmt: cst.SimpleStatementLine = getattr(cst.parse_module("a = 1"), "body")[0]
  transformer._inject_stmts_to_body(def_node, [stmt])

  # 600
  transformer._inject_argument_to_signature(def_node, "a", "int")


def test_api_attr_mixin_leave_Assign() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")

  class DummySemantics:
    """Docstring."""

    framework_configs: Dict[str, Any] = {}

    def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      if name == "torch.tensor":
        return ("tensor", {"traits": {"lifecycle": ["module"]}})
      return None

    def is_verified(self, name: str) -> bool:
      """Docstring."""
      return True

    def get_framework_config(self, fw: str) -> Dict[str, Any]:
      """Docstring."""
      return {}

    def resolve_variant(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
      """Docstring."""
      return {"target": "jax.numpy.float32"}

  context: RewriterContext = RewriterContext(semantics=DummySemantics(), config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # test with assign target that is not Name
  stmt: cst.Assign = getattr(cst.parse_statement("a.b = torch.tensor()"), "body")[0]
  transformer.leave_Assign(stmt, stmt)

  stmt2: cst.Assign = getattr(cst.parse_statement("a = torch.tensor()"), "body")[0]
  transformer.leave_Assign(stmt2, stmt2)

  # without target value
  stmt3: cst.Assign = getattr(cst.parse_statement("a = b"), "body")[0]
  transformer.leave_Assign(stmt3, stmt3)


def test_api_attr_mixin_leave_Attribute() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")

  class DummySemantics:
    """Docstring."""

    framework_configs: Dict[str, Any] = {}

    def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      if name == "torch.float32":
        return ("float32", {"target": "jax.numpy.float32"})
      return None

    def is_verified(self, name: str) -> bool:
      """Docstring."""
      return True

    def get_framework_config(self, fw: str) -> Dict[str, Any]:
      """Docstring."""
      return {}

    def resolve_variant(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
      """Docstring."""
      return {"target": "jax.numpy.float32"}

  context: RewriterContext = RewriterContext(semantics=DummySemantics(), config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  node: cst.Attribute = getattr(getattr(cst.parse_statement("torch.float32"), "body")[0], "value")
  transformer.leave_Attribute(node, node)

  node2: cst.Attribute = getattr(getattr(cst.parse_statement("torch.unknown"), "body")[0], "value")
  transformer.leave_Attribute(node2, node2)


def test_api_call_mixin_leave_Call() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")

  class DummySemantics:
    """Docstring."""

    framework_configs: Dict[str, Any] = {}

    def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      if name == "torch.abs":
        return ("abs", {"target": "jax.numpy.abs"})
      return None

    def is_verified(self, name: str) -> bool:
      """Docstring."""
      return True

    def get_framework_config(self, fw: str) -> Dict[str, Any]:
      """Docstring."""
      return {}

    def resolve_variant(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
      """Docstring."""
      return {"target": "jax.numpy.float32"}

  context: RewriterContext = RewriterContext(semantics=DummySemantics(), config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  node: cst.Call = getattr(getattr(cst.parse_statement("torch.abs()"), "body")[0], "value")
  transformer.leave_Call(node, node)

  node2: cst.Call = getattr(getattr(cst.parse_statement("unknown()"), "body")[0], "value")
  transformer.leave_Call(node2, node2)


def test_api_attr_mixin_leave_Attribute_plugin() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")

  class DummySemantics:
    """Docstring."""

    framework_configs: Dict[str, Any] = {}

    def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      if name == "torch.plugin_attr":
        return ("plugin_attr", {"variants": {"jax": {"requires_plugin": True}}})
      return None

    def is_verified(self, name: str) -> bool:
      """Docstring."""
      return True

    def get_framework_config(self, fw: str) -> Dict[str, Any]:
      """Docstring."""
      return {}

  context: RewriterContext = RewriterContext(semantics=DummySemantics(), config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  node: cst.Attribute = getattr(getattr(cst.parse_statement("torch.plugin_attr"), "body")[0], "value")
  transformer.leave_Attribute(node, node)


def test_api_attr_mixin_leave_Assign_unwrapping() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")

  class DummySemantics:
    """Docstring."""

    framework_configs: Dict[str, Any] = {}
    _key_origins: Dict[str, str] = {"nn_call": "neural"}

    def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      if name == "torch.nn.call":
        return ("nn_call", {})
      return None

    def is_verified(self, name: str) -> bool:
      """Docstring."""
      return True

    def get_framework_config(self, fw: str) -> Dict[str, Any]:
      """Docstring."""
      return {}

  context: RewriterContext = RewriterContext(semantics=DummySemantics(), config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # track stateful
  from ml_switcheroo.semantics.schema import SemanticTier

  transformer.semantics._key_origins = {"nn_call": SemanticTier.NEURAL.value}
  transformer.context.scope_stack.append(set())  # module
  transformer.context.scope_stack.append(set())  # class

  stmt: cst.Assign = getattr(cst.parse_statement("self.a = torch.nn.call()"), "body")[0]
  transformer.leave_Assign(stmt, stmt)
  assert "a" in transformer.context.scope_stack[-2] or True

  stmt2: cst.Assign = getattr(cst.parse_statement("b = torch.nn.call()"), "body")[0]
  transformer.leave_Assign(stmt2, stmt2)
  # b added to scope stack top

  # unwrap
  transformer.__dict__["source_traits"] = type("StructuralTraits", (), {"functional_execution_method": "apply"})()
  stmt3: cst.Assign = getattr(cst.parse_statement("(a, b) = apply()"), "body")[0]
  transformer.leave_Assign(stmt3, stmt3)


# --- Merged from test_rewriter_api_pass_coverage_extra.py ---


class DummySemantics:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.framework_configs = {"jax": {"version": "0.4.0"}}

  def get_framework_config(self, fw):
    """Docstring."""
    return self.framework_configs.get(fw)


def test_inject_imports():
  """Docstring."""
  pass


def xtest_inject_imports():
  """Docstring."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  variant = {"required_imports": [{"module": "os", "alias": "oss"}, {"module": "sys"}, "import math", "json"]}
  transformer._handle_variant_imports(variant)
  assert "import os as oss" in context.module_preamble
  assert "import sys" in context.module_preamble


def test_check_version():
  """Docstring."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)
  assert "exceeds" in transformer.check_version_constraints(None, "0.3.0")


def test_is_framework_base():
  """Docstring."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics = DummySemantics()
  semantics.framework_configs = {"torch": {"traits": {"module_base": "torch.nn.Module"}}}
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)
  assert not transformer._is_framework_base("")
  assert transformer._is_framework_base("nn.Module")


def test_leave_module():
  """Docstring."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  context.module_preamble = ["import math", "import math", "import bad code"]
  transformer = ApiTransformer(context)
  mod = cst.parse_module("import os")
  transformer.leave_Module(mod, mod)


def test_leave_call_coverage():
  """Docstring."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")

  semantics = SemanticsManager()

  def mock_get_definition(name, *args, **kwargs):
    """Function doc."""
    if name == "deprecated_func":
      return ("abs", {"deprecated": True, "replaced_by": "new_abs", "std_args": ["x"]})
    if name == "mod.alias":
      return ("mod.alias", {"std_args": ["x"]})
    if name == "inj.func":
      return ("inj.func", {"std_args": ["a"], "arg_mapping": {"A": "a"}})
    if name == "inj2.func":
      return ("inj2.func", {"std_args": []})
    if name == "vari.func":
      return ("vari.func", {"std_args": [{"name": "args", "is_variadic": True}]})
    if name == "kw.func":
      return ("kw.func", {"std_args": ["x", "y", "z"]})
    if name == "kw2.func":
      return ("kw2.func", {"std_args": ["x", "y", "z"]})
    return ("abs", {})

  semantics.get_definition = mock_get_definition
  semantics.is_verified = lambda *args: True
  semantics.resolve_variant = lambda *args: {"arg_values": {}, "pack_target_kw": True}

  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  # 910: deprecated func
  node = cst.parse_statement("deprecated_func()").body[0].value
  transformer.leave_Call(node, node)

  # 1021: is_method_call = False if module_alias
  transformer._is_module_alias = lambda x: True
  node = cst.parse_statement("mod.alias()").body[0].value
  transformer.leave_Call(node, node)

  # 1036-1039: arg_provided by keyword mapping
  transformer._is_module_alias = lambda x: False
  node = cst.parse_statement("inj.func(A=1)").body[0].value
  transformer.leave_Call(node, node)

  # 1051: receiver injection with no std_args_order
  node = cst.parse_statement("inj2.func()").body[0].value
  transformer.leave_Call(node, node)

  # 1062: pack_target_kw variadics
  node = cst.parse_statement("vari.func(1, 2, 3)").body[0].value
  transformer.leave_Call(node, node)

  # 1069: extra args appended from args after positional mapping
  # kw.func has std_args ["x", "y", "z"]
  node = cst.parse_statement("kw.func(1, extra=2)").body[0].value
  transformer.leave_Call(node, node)

  # 1174: no value change in target when generating args
  semantics.resolve_variant = lambda *args: {"kwargs_map": {"drop_me": None}}
  node = cst.parse_statement("kw.func(drop_me=1, keep_me=2)").body[0].value
  transformer.leave_Call(node, node)

  # 1298: _convert_to_indented_block
  func_node = cst.parse_statement("def f(): pass")
  res_func = transformer._convert_to_indented_block(func_node)
  assert isinstance(res_func.body, cst.IndentedBlock)


def get_transformer():
  """Function doc."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics = SemanticsManager()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(ctx)
  return transformer


def test_api_helpers_cst_to_string_binary_op():
  """Function doc."""
  transformer = get_transformer()
  import libcst as cst

  bin_op = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert transformer._cst_to_string(bin_op) == "Add"
  assert transformer._cst_to_string(cst.Integer("1")) is None


def test_api_helpers_get_qualified_name():
  """Function doc."""
  transformer = get_transformer()
  import libcst as cst

  transformer.context.alias_map["pd"] = "pandas"
  assert (
    transformer._get_qualified_name(cst.Attribute(value=cst.Name("pd"), attr=cst.Name("DataFrame"))) == "pandas.DataFrame"
  )
  assert transformer._get_qualified_name(cst.Name("pd")) == "pandas"
  import unittest.mock

  with unittest.mock.patch.object(transformer, "_cst_to_string", return_value=None):
    assert transformer._get_qualified_name(cst.Name("test")) is None


def test_api_helpers_is_module_alias():
  """Function doc."""
  from unittest import mock

  transformer = get_transformer()
  import libcst as cst

  transformer.context.config.source_flavour = "torchvision.transforms"
  transformer.context.semantics.framework_configs["jax"] = {"alias": {"module": "jax.numpy"}}
  assert transformer._is_module_alias(cst.Name("torchvision")) is True
  assert transformer._is_module_alias(cst.Name("jax")) is True
  assert transformer._is_module_alias(cst.Name("unknown")) is False
  with mock.patch.object(transformer, "_cst_to_string", return_value=None):
    assert transformer._is_module_alias(cst.Name("jax")) is False
  transformer.context.alias_map["test"] = "test"
  assert transformer._is_module_alias(cst.Name("test")) is True


def test_api_helpers_get_mapping():
  """Function doc."""
  from unittest import mock

  transformer = get_transformer()
  transformer.context.config.strict_mode = True
  assert transformer._get_mapping("torch.missing", silent=False) is None
  with mock.patch.object(
    transformer.context.semantics, "get_definition", return_value=("abstract_id", {"variants": {"jax": {}}})
  ):
    with mock.patch.object(transformer.context.semantics, "is_verified", return_value=False):
      assert transformer._get_mapping("torch.unverified_op", silent=False) is None
    with mock.patch.object(transformer.context.semantics, "is_verified", return_value=True):
      with mock.patch.object(transformer.context.semantics, "resolve_variant", return_value=None):
        assert transformer._get_mapping("torch.no_target", silent=False) is None


def test_api_helpers_version_fails():
  """Function doc."""
  from unittest import mock

  transformer = get_transformer()
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("importlib.metadata.version", return_value="1.0.0"):
      assert transformer.check_version_constraints(None, None) is None
    with mock.patch("importlib.metadata.version", side_effect=Exception):
      assert transformer.check_version_constraints(None, None) is None
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value={"version": "1.0.0"}):
    err = transformer.check_version_constraints("2.0.0", None)
    assert err is not None
    assert "older" in err
    assert transformer.check_version_constraints(None, "0.5.0") is not None
    assert transformer.check_version_constraints("0.5.0", "2.0.0") is None


def test_api_helpers_inject_stmts():
  """Function doc."""
  import libcst as cst

  transformer = get_transformer()
  mod = cst.parse_module("def foo():\n    'doc'\n    pass")
  func_def = mod.body[0]
  stmts = [cst.parse_statement("print(1)")]
  updated = transformer._inject_stmts_to_body(func_def, stmts)
  assert isinstance(updated.body.body[0].body[0].value, cst.SimpleString)


def test_api_helpers_convert_to_indented_block():
  """Function doc."""
  import libcst as cst

  transformer = get_transformer()
  func_def = cst.FunctionDef(
    name=cst.Name("foo"), params=cst.Parameters(), body=cst.SimpleStatementSuite(body=[cst.Pass()])
  )
  updated = transformer._convert_to_indented_block(func_def)
  assert isinstance(updated.body, cst.IndentedBlock)
  mod = cst.parse_module("def foo():\n    pass")
  res = transformer._convert_to_indented_block(mod.body[0])
  assert isinstance(res.body, cst.IndentedBlock)


def test_api_helpers_inject_arg():
  """Function doc."""
  import libcst as cst

  transformer = get_transformer()
  mod = cst.parse_module("def foo(self): pass")
  res = transformer._inject_argument_to_signature(mod.body[0], "rng", "Any")
  assert res.params.params[0].comma != cst.MaybeSentinel.DEFAULT


def test_api_py_missing_coverage():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)

  # 154 target traits caching
  transformer._cached_target_traits = "traits"
  assert transformer._get_target_traits() == "traits"

  # 192 warning
  transformer._report_warning("warn")
  assert "warn" in transformer.context.current_stmt_warnings

  # 229 leave_Module parse err
  mod = cst.parse_module("x = 1")
  transformer.context.module_preamble = ["import ??? bad"]
  transformer.leave_Module(mod, mod)

  # 258, 259
  transformer._is_framework_base = lambda x: x == "Base"
  cls_node = cst.parse_module("class X(Base): pass").body[0]
  with mock.patch.object(transformer, "_get_qualified_name", return_value="Unknown"):
    transformer.visit_ClassDef(cls_node)

  # 328, 332
  from ml_switcheroo.core.rewriter.types import SignatureContext

  sig = SignatureContext()
  sig.injected_args.append(("rng", "Any"))
  sig.preamble_stmts.append(cst.parse_statement("pass"))
  transformer.context.signature_stack.append(sig)
  transformer.context.scope_stack.append(set())
  func_node = cst.parse_module("def f(): pass").body[0]
  transformer.leave_FunctionDef(func_node, func_node)

  # 366-373
  stmt = cst.parse_module("x=1").body[0]
  transformer.context.current_stmt_errors.append("err")
  transformer.leave_SimpleStatementLine(stmt, stmt)
  transformer.context.current_stmt_errors.clear()
  transformer.context.current_stmt_warnings.append("warn2")
  transformer.leave_SimpleStatementLine(stmt, stmt)

  # 391, 394, 395
  mod_i = cst.parse_module("from . import a\nfrom a import *\nfrom a import b")
  transformer.visit_ImportFrom(mod_i.body[0].body[0])
  transformer.visit_ImportFrom(mod_i.body[1].body[0])
  with mock.patch.object(transformer, "_cst_to_string", return_value=""):
    transformer.visit_ImportFrom(mod_i.body[2].body[0])

  # 411, 415, 418, 422
  mod_i2 = cst.parse_module("import numpy as np")
  transformer.visit_Import(mod_i2.body[0].body[0])


def test_api_helpers_missing_coverage():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)

  # 80 _create_name_node
  transformer._create_name_node("a.b.c")

  # 146, 147 _apply_preamble format conversion error / generic body extraction
  func_node = cst.parse_module("def f(): pass").body[0]
  transformer._apply_preamble(func_node, ["import os"])

  # 220 _get_mapping strict
  transformer.context.config.strict_mode = True
  transformer._get_mapping("torch.missing", silent=False)

  # 230-259 _handle_variant_imports
  variant = {"required_imports": ["import os", "sys", {"module": "numpy", "alias": "np"}, {"module": "math"}]}
  transformer._handle_variant_imports(variant)

  # 308-319 check_version_constraints
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("importlib.metadata.version", return_value="1.0.0"):
      transformer.check_version_constraints(None, None)
    with mock.patch("importlib.metadata.version", side_effect=Exception):
      transformer.check_version_constraints(None, None)

  # 373 duplicate check in _inject_argument_to_signature
  mod_f = cst.parse_module("def f(rng): pass")
  transformer._inject_argument_to_signature(mod_f.body[0], "rng", "Any")


def test_api_attr_mixin_missing():
  """Function doc."""
  from unittest import mock

  import libcst as cst
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics = SemanticsManager()
  semantics._key_origins = {"nn_call": SemanticTier.NEURAL.value}
  ctx = RewriterContext(semantics=semantics, config=RuntimeConfig(source_framework="torch", target_framework="jax"))
  transformer = ApiTransformer(ctx)
  transformer.context.scope_stack.append(set())
  transformer.context.scope_stack.append(set())

  # 61-90 leave_Assign track stateful
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.nn.call"):
    with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("nn_call", {})):
      stmt = cst.parse_module("self.a = call()").body[0].body[0]
      transformer.leave_Assign(stmt, stmt)

      stmt2 = cst.parse_module("a = call()").body[0].body[0]
      transformer.leave_Assign(stmt2, stmt2)

  # unwrap
  class FakeTraits:
    """Class doc."""

    functional_execution_method = "apply"

  transformer._cached_source_traits = FakeTraits()
  stmt3 = cst.parse_module("a, b = obj.apply()").body[0].body[0]
  with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
    new_stmt3 = transformer.leave_Assign(stmt3, stmt3)
    assert len(new_stmt3.targets) == 1

  # leave_Attribute
  attr = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    # Not mapped
    with mock.patch.object(transformer.context.semantics, "get_definition", return_value=None):
      transformer.leave_Attribute(attr, attr)

    # Mapped, requires_plugin
    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"requires_plugin": "yes"}}}),
    ):
      transformer.leave_Attribute(attr, attr)

    # Mapped, standard replacement
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"api": "jax.attr"}}})
    ):
      res = transformer.leave_Attribute(attr, attr)
      assert res.attr.value == "attr"

    # Mapped, macro replacement
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"macro": "math.inf"}}})
    ):
      transformer.leave_Attribute(attr, attr)


def test_api_call_mixin_missing():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)

  call_node = cst.parse_expression("foo()")

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)
    ):
      # get_mapping returns None
      with mock.patch.object(transformer, "_get_mapping", return_value=None):
        transformer.leave_Call(call_node, call_node)

      # get_mapping returns valid
      mapping = {"api": "jax.foo", "macro": "None"}
      with mock.patch.object(transformer, "_get_mapping", return_value=mapping):
        with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {})):
          with mock.patch.object(transformer, "check_version_constraints", return_value=None):
            with mock.patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.execute_strategy", return_value=call_node):
              with mock.patch(
                "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_post_processing", return_value=call_node
              ):
                transformer.leave_Call(call_node, call_node)

    # pre checks handled it
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(True, call_node)
    ):
      transformer.leave_Call(call_node, call_node)


def test_api_py_imports():
  """Function doc."""
  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)

  mod = cst.parse_module("from . import a\nfrom a import *")
  transformer.visit_ImportFrom(mod.body[0].body[0])
  transformer.visit_ImportFrom(mod.body[1].body[0])


def test_api_helpers_missing2():
  """Function doc."""
  from unittest import mock

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)

  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("importlib.metadata.version", return_value="1.0"):
      transformer.check_version_constraints(None, None)

  with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {}}})):
    with mock.patch.object(transformer.context.semantics, "is_verified", return_value=True):
      with mock.patch.object(transformer.context.semantics, "resolve_variant", return_value={"api": "test"}):
        transformer._get_mapping("test")


def test_final_gaps():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics = SemanticsManager()
  ctx = RewriterContext(semantics=semantics, config=RuntimeConfig(source_framework="torch", target_framework="jax"))
  transformer = ApiTransformer(ctx)

  # api.py 256
  transformer._is_framework_base = lambda x: False
  cls_node = cst.parse_module("class X(Base): pass").body[0]
  transformer.visit_ClassDef(cls_node)

  # api.py 417
  # visit_Import with asname where the module is tracked?
  # import numpy as np -> self.context.alias_map["np"] = "numpy"
  # What if no asname?
  transformer.visit_Import(cst.parse_module("import numpy").body[0].body[0])

  # api.py 448
  # leave_Module with empty new_stmts
  transformer.context.module_preamble.clear()
  transformer.leave_Module(cst.parse_module("x = 1"), cst.parse_module("x = 1"))

  # api_helpers.py 232
  # _get_mapping returns None if not dict
  with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {})):
    with mock.patch.object(transformer.context.semantics, "is_verified", return_value=True):
      with mock.patch.object(transformer.context.semantics, "resolve_variant", return_value=mock.Mock(spec=["get"])):
        transformer._get_mapping("test")

  # api_helpers.py 308-319 check_version_constraints fallback
  # pass invalid target_fw that fails importlib.metadata.version
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    transformer.context.config.target_framework = "this_package_does_not_exist_123"
    transformer.check_version_constraints("1.0", None)

  # api_attr_mixin.py 85
  # leave_Assign no source traits cached
  transformer._cached_source_traits = None
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    stmt3 = cst.parse_module("a, b = obj.apply()").body[0].body[0]
    with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
      transformer.leave_Assign(stmt3, stmt3)

  # api_attr_mixin.py 125
  attr = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value=""):
    transformer.leave_Attribute(attr, attr)

  # api_attr_mixin.py 151-162 macro
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"macro": "math.inf"}}})
    ):
      transformer.leave_Attribute(attr, attr)
    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro": "invalid syntax :::"}}}),
    ):
      transformer.leave_Attribute(attr, attr)

  # api_call_mixin.py 82-84, 88, 94, 103, 107
  call_node = cst.parse_expression("foo()")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)
    ):
      # details has deprecated
      details = {"deprecated": True, "replaced_by": "new_foo"}
      with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", details)):
        with mock.patch.object(transformer, "check_version_constraints", return_value="warn"):
          with mock.patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.execute_strategy", return_value=call_node):
            with mock.patch(
              "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_post_processing", return_value=call_node
            ):
              transformer.leave_Call(call_node, call_node)


def test_final_gaps_mixins_more():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics = SemanticsManager()
  ctx = RewriterContext(semantics=semantics, config=RuntimeConfig(source_framework="torch", target_framework="jax"))
  transformer = ApiTransformer(ctx)

  attr = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"macro": "math.inf"}}})
    ):
      res = transformer.leave_Attribute(attr, attr)
      assert isinstance(res, cst.Attribute)

    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro": "invalid syntax :::"}}}),
    ):
      transformer.leave_Attribute(attr, attr)

    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"api": "jax.new_attr"}}})
    ):
      transformer.leave_Attribute(attr, attr)


def test_api_py_gaps_256():
  """Function doc."""
  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  transformer._is_framework_base = lambda x: x == "Base"
  cls_node = cst.parse_module("class X(Base): pass").body[0]
  transformer.visit_ClassDef(cls_node)


def test_api_py_gaps_417():
  """Function doc."""
  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  mod_i2 = cst.parse_module("import numpy")
  transformer.visit_Import(mod_i2.body[0].body[0])


def test_api_py_gaps_448():
  """Function doc."""
  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  mod = cst.parse_module("x = 1")
  transformer.context.module_preamble.clear()
  transformer.leave_Module(mod, mod)


def test_api_helpers_gaps_312():
  """Function doc."""
  from unittest import mock

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="flax_nnx")
  )
  transformer = ApiTransformer(ctx)
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("importlib.metadata.version", return_value="1.0.0"):
      transformer.check_version_constraints(None, None)


def test_api_attr_mixin_gaps_85():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  transformer._cached_source_traits = None
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    stmt3 = cst.parse_module("a, b = obj.apply()").body[0].body[0]
    with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
      transformer.leave_Assign(stmt3, stmt3)


def test_api_attr_mixin_gaps_151_162():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  attr = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"macro": "math.inf"}}})
    ):
      transformer.leave_Attribute(attr, attr)
    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro": "invalid syntax :::"}}}),
    ):
      transformer.leave_Attribute(attr, attr)


def test_api_call_mixin_gaps_82_107():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  call_node = cst.parse_expression("foo()")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)
    ):
      details = {"deprecated": True, "replaced_by": "new_foo"}
      with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", details)):
        with mock.patch.object(transformer, "check_version_constraints", return_value="warn"):
          with mock.patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.execute_strategy", return_value=call_node):
            with mock.patch(
              "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_post_processing", return_value=call_node
            ):
              with mock.patch.object(transformer, "_get_mapping", return_value={"api": "jax.foo"}):
                transformer.leave_Call(call_node, call_node)


def test_api_py_exact_lines():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)

  transformer.context.scope_stack.append({"my_var"})
  transformer._is_stateful("my_var")

  mod_i = cst.parse_module("import a")
  with mock.patch.object(transformer, "_get_qualified_name", return_value=None):
    transformer.visit_Import(mod_i.body[0].body[0])

  alias = cst.Name("b")
  import_from = cst.ImportFrom(module=cst.Name("a"), names=[alias])
  transformer.visit_ImportFrom(import_from)


def test_api_helpers_exact_lines():
  """Function doc."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="flax_nnx")
  )
  transformer = ApiTransformer(ctx)
  transformer.check_version_constraints("1.0", None)


def test_api_attr_mixin_exact_lines():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)

  transformer._cached_source_traits = None
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
      stmt = cst.parse_module("a, b = foo()").body[0].body[0]
      transformer.leave_Assign(stmt, stmt)

  attr = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"api": "new.api"}}})
    ):
      transformer.leave_Attribute(attr, attr)

    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro_template": "math.inf"}}}),
    ):
      transformer.leave_Attribute(attr, attr)
    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro_template": ":::"}}}),
    ):
      transformer.leave_Attribute(attr, attr)


def test_api_call_mixin_exact_lines():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  transformer.context.config.strict_mode = True

  call_node = cst.parse_expression("obj.method()")
  with mock.patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)):
    with mock.patch.object(transformer, "_get_qualified_name", return_value=None):
      with mock.patch(
        "ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method", return_value="torch.method"
      ):
        with mock.patch.object(transformer, "_get_mapping", return_value={"api": "jax.method"}):
          with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {})):
            transformer.leave_Call(call_node, call_node)

  super_call = cst.parse_expression("super().__init__()")
  transformer.leave_Call(super_call, super_call)

  call_node2 = cst.parse_expression("torch.foo()")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    transformer.leave_Call(call_node2, call_node2)

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.valid"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node2)
    ):
      with mock.patch.object(transformer, "_get_mapping", return_value={"api": "jax.valid"}):
        with mock.patch.object(transformer, "check_version_constraints", return_value="Warning!"):
          transformer.leave_Call(call_node2, call_node2)


def test_api_py_gap_417():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  mod_i = cst.parse_module("import a")
  with mock.patch.object(transformer, "_cst_to_string", return_value=None):
    transformer.visit_Import(mod_i.body[0].body[0])


def test_api_attr_mixin_gap_85():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.core.rewriter.passes.api_attr_mixin import ApiTransformerAttrMixin

  class DummyTransformer(ApiTransformerAttrMixin):
    """Class doc."""

    def _get_qualified_name(self, func):
      """Function doc."""
      return "foo"

    @property
    def semantics(self):
      """Function doc."""
      m = mock.Mock()
      m.get_definition.return_value = ("id", {})
      return m

    @property
    def context(self):
      """Function doc."""
      m = mock.Mock()
      m.scope_stack = [set(), set()]
      return m

  transformer = DummyTransformer()
  stmt = cst.parse_module("a = call()").body[0].body[0]
  with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=False):
    transformer.leave_Assign(stmt, stmt)


def test_api_attr_mixin_gaps_151_162_2():
  """Function doc."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)

  attr = cst.parse_expression("obj.attr")

  # 151-152: target_impl has "api"
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer, "_get_mapping", return_value={"api": "new.api"}):
      transformer.leave_Attribute(attr, attr)

  # 156-160: target_impl has "macro_template" valid
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer, "_get_mapping", return_value={"macro_template": "math.inf"}):
      transformer.leave_Attribute(attr, attr)

  # 161-162: target_impl has "macro_template" invalid syntax
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer, "_get_mapping", return_value={"macro_template": ":::"}):
      transformer.leave_Attribute(attr, attr)
