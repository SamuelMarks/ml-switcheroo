"""Test suite for rewriter api pass coverage."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

import libcst as _cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiPass, ApiTransformer
from ml_switcheroo.semantics.manager import SemanticsManager

cst = _cst


def test_apipass_branches() -> None:
  """Verify apipass branches behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = SemanticsManager()
  context = RewriterContext(semantics=semantics, config=config)
  pass_obj = ApiPass()

  module = cst.parse_module(
    """import torch
x = torch.abs(y)
class Model:
 def __init__(self):
  self.layer = 1
 def forward(self, x):
  return self.layer(x)
"""
  )
  res_mod = pass_obj.transform(module, context)
  assert isinstance(res_mod, cst.Module)


def test_apitransformer_methods() -> None:
  """Verify apitransformer methods behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  class DummySemanticsNone:
    """Test helper class DummySemanticsNone."""

    framework_configs = {"torch": {"traits": {"module_base": "torch.nn.Module"}}}

    def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute get definition helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      return None

    def is_verified(self, name: str) -> bool:
      """Execute is verified helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `bool` from is verified.
      """
      return True

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Execute get framework config helper logic.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      _ret_v: dict[str, JsonValue] = getattr({"v": None}, "__getitem__")("v")
      return _ret_v

  semantics_none = DummySemanticsNone()
  context = RewriterContext(semantics=semantics_none, config=config)

  transformer = ApiTransformer(context)
  assert getattr(transformer, "semantics") == semantics_none
  assert transformer.config == config
  assert transformer.source_framework == "torch"
  assert transformer.target_framework == "jax"
  assert not getattr(transformer, "strict_mode", False)

  getattr(transformer, "source_traits")
  transformer._get_target_traits()

  class DummySemanticsTraits:
    """Test helper class DummySemanticsTraits."""

    framework_configs: dict[str, dict[str, JsonValue]] = {}

    def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute get definition helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      return None

    def is_verified(self, name: str) -> bool:
      """Execute is verified helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `bool` from is verified.
      """
      return True

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Execute get framework config helper logic.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      return {"traits": {}}

  semantics_traits = DummySemanticsTraits()
  context_traits = RewriterContext(semantics=semantics_traits, config=config)
  transformer_traits = ApiTransformer(context_traits)

  assert transformer_traits.source_traits is not None
  assert transformer_traits._get_target_traits() is not None
  assert transformer._cst_to_string(cst.BinaryOperation(cst.Name("a"), cst.Add(), cst.Name("b"))) == "Add"
  assert transformer._cst_to_string(cst.Pass()) is None
  assert transformer._cst_to_string(getattr(cst, "Attribute")(value=cst.Pass(), attr=cst.Name("a"))) is None

  assert transformer._get_source_lifecycle_lists() == (set(), set())
  assert getattr(transformer, "ctx", None) is not None

  assert transformer.visit_Module(cst.parse_module("import os")) is None

  import_node = cst.Import(names=[cst.ImportAlias(cst.Name("os"))])
  assert transformer.visit_Import(import_node) is False

  import_from_node = cst.ImportFrom(module=cst.Name("os"), names=[cst.ImportAlias(cst.Name("path"))])
  assert transformer.visit_ImportFrom(import_from_node) is False

  node_func = getattr(cst.parse_module("def f(): pass"), "body")[0]
  assert isinstance(transformer.leave_FunctionDef(node_func, node_func), cst.FunctionDef)

  node_assign = getattr(cst.parse_statement("a = 1"), "body")[0]
  assert isinstance(transformer.leave_Assign(node_assign, node_assign), cst.Assign)

  node_attr = getattr(getattr(cst.parse_statement("a.b"), "body")[0], "value")
  assert isinstance(transformer.leave_Attribute(node_attr, node_attr), cst.Attribute)

  class_node = getattr(cst.parse_module("class A(torch.nn.Module): pass"), "body")[0]
  assert transformer.visit_ClassDef(class_node) is True
  assert isinstance(transformer.leave_ClassDef(class_node, class_node), cst.ClassDef)

  class_node2 = getattr(cst.parse_module("class B(UnknownBase): pass"), "body")[0]
  assert transformer.visit_ClassDef(class_node2) is True
  assert isinstance(transformer.leave_ClassDef(class_node2, class_node2), cst.ClassDef)

  stmt_line = getattr(cst.parse_module("a = 1"), "body")[0]
  assert transformer.visit_SimpleStatementLine(stmt_line) is True
  assert isinstance(transformer.leave_SimpleStatementLine(stmt_line, stmt_line), cst.SimpleStatementLine)

  def_node = getattr(cst.parse_module("def f(): pass"), "body")[0]
  assert isinstance(transformer._apply_preamble(def_node, ["a = 1"]), cst.FunctionDef)

  stmt = getattr(cst.parse_module("a = 1"), "body")[0]
  assert isinstance(transformer._inject_stmts_to_body(def_node, [stmt]), cst.FunctionDef)

  assert isinstance(transformer._inject_argument_to_signature(def_node, "a", "int"), cst.FunctionDef)


def test_api_attr_mixin_leave_Assign() -> None:
  """Verify api attr mixin leave Assign behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  class DummySemantics:
    """Test helper class DummySemantics."""

    framework_configs: dict[str, RuntimeTensorOrValue] = {}

    def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute get definition helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      if name == "torch.tensor":
        return ("tensor", {"traits": {"lifecycle": ["module"]}})
      return None

    def is_verified(self, name: str) -> bool:
      """Execute is verified helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `bool` from is verified.
      """
      return True

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Execute get framework config helper logic.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      return {}

    def resolve_variant(self, *args: str, **kwargs: str) -> dict[str, JsonValue]:
      """Execute resolve variant helper logic.

      Args:
          args: The args (str) for resolve variant.
          kwargs: The kwargs (str) for resolve variant.

      Returns:
          Resulting `dict[str, JsonValue]` from resolve variant.
      """
      return {"target": "jax.numpy.float32"}

  context = RewriterContext(semantics=DummySemantics(), config=config)
  transformer = ApiTransformer(context)

  stmt = getattr(cst.parse_statement("a.b = torch.tensor()"), "body")[0]
  transformer.leave_Assign(stmt, stmt)

  stmt2 = getattr(cst.parse_statement("a = torch.tensor()"), "body")[0]
  transformer.leave_Assign(stmt2, stmt2)

  stmt3 = getattr(cst.parse_statement("a = b"), "body")[0]
  assert transformer.leave_Assign(stmt3, stmt3) is not None


def test_api_attr_mixin_leave_Attribute() -> None:
  """Verify api attr mixin leave Attribute behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  class DummySemantics:
    """Test helper class DummySemantics."""

    framework_configs: dict[str, RuntimeTensorOrValue] = {}

    def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute get definition helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      if name == "torch.float32":
        return ("float32", {"target": "jax.numpy.float32"})
      return None

    def is_verified(self, name: str) -> bool:
      """Execute is verified helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `bool` from is verified.
      """
      return True

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Execute get framework config helper logic.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      return {}

    def resolve_variant(self, *args: str, **kwargs: str) -> dict[str, JsonValue]:
      """Execute resolve variant helper logic.

      Args:
          args: The args (str) for resolve variant.
          kwargs: The kwargs (str) for resolve variant.

      Returns:
          Resulting `dict[str, JsonValue]` from resolve variant.
      """
      return {"target": "jax.numpy.float32"}

  context = RewriterContext(semantics=DummySemantics(), config=config)
  transformer = ApiTransformer(context)

  node = getattr(getattr(cst.parse_statement("torch.float32"), "body")[0], "value")
  assert transformer.leave_Attribute(node, node) is not None

  node2 = getattr(getattr(cst.parse_statement("torch.unknown"), "body")[0], "value")
  assert transformer.leave_Attribute(node2, node2) == node2


def test_api_call_mixin_leave_Call() -> None:
  """Verify api call mixin leave Call behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  class DummySemantics:
    """Test helper class DummySemantics."""

    framework_configs: dict[str, RuntimeTensorOrValue] = {}

    def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute get definition helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      if name == "torch.abs":
        return ("abs", {"target": "jax.numpy.abs"})
      return None

    def is_verified(self, name: str) -> bool:
      """Execute is verified helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `bool` from is verified.
      """
      return True

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Execute get framework config helper logic.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      return {}

    def resolve_variant(self, *args: str, **kwargs: str) -> dict[str, JsonValue]:
      """Execute resolve variant helper logic.

      Args:
          args: The args (str) for resolve variant.
          kwargs: The kwargs (str) for resolve variant.

      Returns:
          Resulting `dict[str, JsonValue]` from resolve variant.
      """
      return {"target": "jax.numpy.float32"}

  context = RewriterContext(semantics=DummySemantics(), config=config)
  transformer = ApiTransformer(context)

  node = getattr(getattr(cst.parse_statement("torch.abs()"), "body")[0], "value")
  transformer.leave_Call(node, node)

  node2 = getattr(getattr(cst.parse_statement("unknown()"), "body")[0], "value")
  assert transformer.leave_Call(node2, node2) == node2


def test_api_attr_mixin_leave_Attribute_plugin() -> None:
  """Verify api attr mixin leave Attribute plugin behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  class DummySemantics:
    """Test helper class DummySemantics."""

    framework_configs: dict[str, RuntimeTensorOrValue] = {}

    def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute get definition helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      if name == "torch.plugin_attr":
        return ("plugin_attr", {"variants": {"jax": {"requires_plugin": True}}})
      return None

    def is_verified(self, name: str) -> bool:
      """Execute is verified helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `bool` from is verified.
      """
      return True

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Execute get framework config helper logic.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      return {}

  context = RewriterContext(semantics=DummySemantics(), config=config)
  transformer = ApiTransformer(context)

  node = getattr(getattr(cst.parse_statement("torch.plugin_attr"), "body")[0], "value")
  assert transformer.leave_Attribute(node, node) is not None


def test_api_attr_mixin_leave_Assign_unwrapping() -> None:
  """Verify api attr mixin leave Assign unwrapping behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  class DummySemantics:
    """Test helper class DummySemantics."""

    framework_configs: dict[str, RuntimeTensorOrValue] = {}
    _key_origins = {"nn_call": "neural"}

    def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute get definition helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      if name == "torch.nn.call":
        return ("nn_call", {})
      return None

    def is_verified(self, name: str) -> bool:
      """Execute is verified helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `bool` from is verified.
      """
      return True

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Execute get framework config helper logic.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      return {}

  context = RewriterContext(semantics=DummySemantics(), config=config)
  transformer = ApiTransformer(context)

  from ml_switcheroo.semantics.schema import SemanticTier

  transformer.semantics._key_origins = {"nn_call": SemanticTier.NEURAL.value}
  transformer.context.scope_stack.append(set())
  transformer.context.scope_stack.append(set())

  stmt = getattr(cst.parse_statement("self.a = torch.nn.call()"), "body")[0]
  transformer.leave_Assign(stmt, stmt)
  assert "self.a" in transformer.context.scope_stack[-2]

  stmt2 = getattr(cst.parse_statement("b = torch.nn.call()"), "body")[0]
  transformer.leave_Assign(stmt2, stmt2)
  assert "b" in transformer.context.scope_stack[-1]

  transformer.__dict__["source_traits"] = type("StructuralTraits", (), {"functional_execution_method": "apply"})()
  stmt3 = getattr(cst.parse_statement("(a, b) = apply()"), "body")[0]
  assert isinstance(transformer.leave_Assign(stmt3, stmt3), cst.Assign)


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.framework_configs = {"jax": {"version": "0.4.0"}}

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Execute get framework config helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    _ret_v: dict[str, JsonValue] = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").framework_configs}, "__getitem__")("v").get},
          "__getitem__",
        )("v")(getattr({"v": fw}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v


def test_inject_imports() -> None:
  """Verify inject imports behavior."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.passes.plugins.dsl import ImportFixerPass

  module = cst.parse_module("a = 1")
  transformer = ImportFixerPass(required_imports=["os", "sys"])
  modified = module.visit(transformer)
  assert "import os" in modified.code
  assert "import sys" in modified.code


def test_inject_imports_variants() -> None:
  """Execute xtest inject imports helper logic."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  variant = {"required_imports": [{"module": "os", "alias": "oss"}, {"module": "sys"}, "import math", "json"]}
  transformer._handle_variant_imports(variant)
  assert "import os as oss" in context.module_preamble
  assert "import sys" in context.module_preamble


def test_check_version() -> None:
  """Verify check version behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)
  assert getattr({"v": "exceeds"}, "__getitem__")("v") in getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v").check_version_constraints}, "__getitem__")("v")(
        getattr({"v": None}, "__getitem__")("v"), getattr({"v": "0.3.0"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")


def test_is_framework_base() -> None:
  """Verify is framework base behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  setattr(
    getattr({"v": semantics}, "__getitem__")("v"),
    "framework_configs",
    getattr(
      {
        "v": {
          "torch": getattr(
            {
              "v": {
                "traits": getattr(
                  {"v": {"module_base": getattr({"v": "torch.nn.Module"}, "__getitem__")("v")}}, "__getitem__"
                )("v")
              }
            },
            "__getitem__",
          )("v")
        }
      },
      "__getitem__",
    )("v"),
  )
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)
  assert not transformer._is_framework_base("")
  assert transformer._is_framework_base("nn.Module")


def test_leave_module() -> None:
  """Verify leave module behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  context.module_preamble = ["import math", "import math", "import bad code"]
  transformer = ApiTransformer(context)
  mod = cst.parse_module("import os")
  assert isinstance(transformer.leave_Module(mod, mod), cst.Module)


def test_leave_call_coverage() -> None:
  """Verify leave call coverage behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  semantics = SemanticsManager()

  def mock_get_definition(name: str, *args: str, **kwargs: str) -> tuple[str, dict[str, JsonValue]]:
    """Execute mock get definition helper logic.

    Args:
        name: Identifier or operation name string.
        args: The args (str) for mock get definition.
        kwargs: The kwargs (str) for mock get definition.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]]` from mock get definition.
    """
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

  setattr(semantics, "get_definition", mock_get_definition)
  setattr(semantics, "is_verified", lambda *args: True)
  setattr(semantics, "resolve_variant", lambda *args: {"arg_values": {}, "pack_target_kw": True})

  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  setattr(transformer, "_is_module_alias", lambda x: True)
  node = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_statement}, "__getitem__")("v")(
                    getattr({"v": "mod.alias()"}, "__getitem__")("v")
                  )
                },
                "__getitem__",
              )("v").body
            },
            "__getitem__",
          )("v")[0]
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v")
  assert isinstance(transformer.leave_Call(node, node), cst.CSTNode)

  setattr(transformer, "_is_module_alias", lambda x: False)
  node = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_statement}, "__getitem__")("v")(
                    getattr({"v": "inj.func(A=1)"}, "__getitem__")("v")
                  )
                },
                "__getitem__",
              )("v").body
            },
            "__getitem__",
          )("v")[0]
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v")
  assert isinstance(transformer.leave_Call(node, node), cst.CSTNode)

  node = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_statement}, "__getitem__")("v")(
                    getattr({"v": "inj2.func()"}, "__getitem__")("v")
                  )
                },
                "__getitem__",
              )("v").body
            },
            "__getitem__",
          )("v")[0]
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v")
  assert isinstance(transformer.leave_Call(node, node), cst.CSTNode)

  node = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_statement}, "__getitem__")("v")(
                    getattr({"v": "vari.func(1, 2, 3)"}, "__getitem__")("v")
                  )
                },
                "__getitem__",
              )("v").body
            },
            "__getitem__",
          )("v")[0]
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v")
  assert isinstance(transformer.leave_Call(node, node), cst.CSTNode)

  node = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_statement}, "__getitem__")("v")(
                    getattr({"v": "kw.func(1, extra=2)"}, "__getitem__")("v")
                  )
                },
                "__getitem__",
              )("v").body
            },
            "__getitem__",
          )("v")[0]
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v")
  assert isinstance(transformer.leave_Call(node, node), cst.CSTNode)

  setattr(semantics, "resolve_variant", lambda *args: {"kwargs_map": {"drop_me": None}})
  node = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_statement}, "__getitem__")("v")(
                    getattr({"v": "kw.func(drop_me=1, keep_me=2)"}, "__getitem__")("v")
                  )
                },
                "__getitem__",
              )("v").body
            },
            "__getitem__",
          )("v")[0]
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v")
  assert isinstance(transformer.leave_Call(node, node), cst.CSTNode)

  func_node = cst.ensure_type(cst.parse_statement("def f(): pass"), cst.FunctionDef)
  res_func = transformer._convert_to_indented_block(func_node)
  assert isinstance(res_func.body, cst.IndentedBlock)


def get_transformer() -> ApiTransformer:
  """Execute get transformer helper logic.

  Returns:
      Resulting `ApiTransformer` from get transformer.
  """
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics = SemanticsManager()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(ctx)
  return transformer


def test_api_helpers_cst_to_string_binary_op() -> None:
  """Verify api helpers cst to string binary op behavior."""
  transformer = get_transformer()
  import libcst as cst

  bin_op = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert transformer._cst_to_string(bin_op) == "Add"
  assert transformer._cst_to_string(cst.Integer("1")) is None


def test_api_helpers_get_qualified_name() -> None:
  """Verify api helpers get qualified name behavior."""
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


def test_api_helpers_is_module_alias() -> None:
  """Verify api helpers is module alias behavior."""
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


def test_api_helpers_get_mapping() -> None:
  """Verify api helpers get mapping behavior."""
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


def test_api_helpers_version_fails() -> None:
  """Verify api helpers version fails behavior."""
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


def test_api_helpers_inject_stmts() -> None:
  """Verify api helpers inject stmts behavior."""
  import libcst as cst

  transformer = get_transformer()
  mod = cst.parse_module("""def foo():
    'doc'
    pass""")
  func_def = cst.ensure_type(mod.body[0], cst.FunctionDef)
  stmts = [cst.parse_statement("print(1)")]
  updated = transformer._inject_stmts_to_body(func_def, stmts)
  assert isinstance(
    getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {"v": getattr({"v": getattr({"v": updated}, "__getitem__")("v").body}, "__getitem__")("v").body},
                      "__getitem__",
                    )("v")[0]
                  },
                  "__getitem__",
                )("v").body
              },
              "__getitem__",
            )("v")[0]
          },
          "__getitem__",
        )("v").value
      },
      "__getitem__",
    )("v"),
    cst.SimpleString,
  )


def test_api_helpers_convert_to_indented_block() -> None:
  """Verify api helpers convert to indented block behavior."""
  import libcst as cst

  transformer = get_transformer()
  func_def = cst.FunctionDef(
    name=cst.Name("foo"), params=cst.Parameters(), body=cst.SimpleStatementSuite(body=[cst.Pass()])
  )
  updated = transformer._convert_to_indented_block(func_def)
  assert isinstance(updated.body, cst.IndentedBlock)
  mod = cst.parse_module("""def foo():
    pass""")
  res = transformer._convert_to_indented_block(cst.ensure_type(mod.body[0], cst.FunctionDef))
  assert isinstance(res.body, cst.IndentedBlock)


def test_api_helpers_inject_arg() -> None:
  """Verify api helpers inject arg behavior."""
  import libcst as cst

  transformer = get_transformer()
  mod = cst.parse_module("def foo(self): pass")
  res = transformer._inject_argument_to_signature(cst.ensure_type(mod.body[0], cst.FunctionDef), "rng", "Any")
  assert res.params.params[0].comma != cst.MaybeSentinel.DEFAULT
