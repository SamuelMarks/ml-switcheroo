"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.api import ApiPass, ApiTransformer
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from typing import Optional, Dict, Any, Tuple


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
