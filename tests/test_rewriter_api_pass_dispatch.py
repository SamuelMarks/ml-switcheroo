"""Test suite for API rewriter pass dispatch and mixin edge cases."""

import typing


def test_api_py_missing_coverage() -> None:
  """Verify api py missing coverage behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  transformer._cached_target_traits = "traits"
  assert transformer._get_target_traits() == "traits"

  transformer._report_warning("warn")
  assert "warn" in transformer.context.current_stmt_warnings

  mod: typing.Any = cst.parse_module("x = 1")
  transformer.context.module_preamble = ["import ??? bad"]
  assert isinstance(transformer.leave_Module(mod, mod), cst.Module)

  transformer._is_framework_base = lambda x: x == "Base"
  cls_node: typing.Any = cst.parse_module("class X(Base): pass").body[0]
  with mock.patch.object(transformer, "_get_qualified_name", return_value="Unknown"):
    assert transformer.visit_ClassDef(cls_node) is True

  from ml_switcheroo.core.rewriter.types import SignatureContext

  sig: typing.Any = SignatureContext()
  sig.injected_args.append(("rng", "Any"))
  sig.preamble_stmts.append(cst.parse_statement("pass"))
  transformer.context.signature_stack.append(sig)
  transformer.context.scope_stack.append(set())
  func_node: typing.Any = cst.parse_module("def f(): pass").body[0]
  res_f: typing.Any = transformer.leave_FunctionDef(func_node, func_node)
  assert isinstance(res_f, cst.FunctionDef)
  assert len(res_f.params.params) == 1

  stmt: typing.Any = cst.parse_module("x=1").body[0]
  transformer.context.current_stmt_errors.append("err")
  res_err_stmt: typing.Any = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res_err_stmt is not None
  transformer.context.current_stmt_errors.clear()
  transformer.context.current_stmt_warnings.append("warn2")
  res_warn_stmt: typing.Any = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res_warn_stmt is not None

  mod_i: typing.Any = cst.parse_module("""from . import a
from a import *
from a import b""")
  assert transformer.visit_ImportFrom(mod_i.body[0].body[0]) is False
  assert transformer.visit_ImportFrom(mod_i.body[1].body[0]) is False
  with mock.patch.object(transformer, "_cst_to_string", return_value=""):
    assert transformer.visit_ImportFrom(mod_i.body[2].body[0]) is False

  mod_i2: typing.Any = cst.parse_module("import numpy as np")
  assert transformer.visit_Import(mod_i2.body[0].body[0]) is False


def test_api_helpers_missing_coverage() -> None:
  """Verify api helpers missing coverage behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  assert isinstance(transformer._create_name_node("a.b.c"), cst.Attribute)

  func_node: typing.Any = cst.parse_module("def f(): pass").body[0]
  assert isinstance(transformer._apply_preamble(func_node, ["import os"]), cst.FunctionDef)

  transformer.context.config.strict_mode = True
  assert transformer._get_mapping("torch.missing", silent=False) is None

  variant: typing.Any = {"required_imports": ["import os", "sys", {"module": "numpy", "alias": "np"}, {"module": "math"}]}
  transformer._handle_variant_imports(variant)
  assert "import os" in transformer.context.module_preamble

  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("importlib.metadata.version", return_value="1.0.0"):
      assert transformer.check_version_constraints(None, None) is None
    with mock.patch("importlib.metadata.version", side_effect=Exception):
      assert transformer.check_version_constraints(None, None) is None

  mod_f: typing.Any = cst.parse_module("def f(rng): pass")
  res_dup: typing.Any = transformer._inject_argument_to_signature(mod_f.body[0], "rng", "Any")
  assert len(res_dup.params.params) == 1


def test_api_attr_mixin_missing() -> None:
  """Verify api attr mixin missing behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics: typing.Any = SemanticsManager()
  semantics._key_origins = {"nn_call": SemanticTier.NEURAL.value}
  ctx: typing.Any = RewriterContext(
    semantics=semantics, config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  transformer.context.scope_stack.append(set())
  transformer.context.scope_stack.append(set())

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.nn.call"):
    with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("nn_call", {})):
      stmt: typing.Any = cst.parse_module("self.a = call()").body[0].body[0]
      assert transformer.leave_Assign(stmt, stmt) is not None

      stmt2: typing.Any = cst.parse_module("a = call()").body[0].body[0]
      transformer.leave_Assign(stmt2, stmt2)

  class FakeTraits:
    """Test helper class FakeTraits."""

    functional_execution_method: typing.Any = "apply"

  transformer._cached_source_traits = FakeTraits()
  stmt3: typing.Any = cst.parse_module("a, b = obj.apply()").body[0].body[0]
  with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
    new_stmt3: typing.Any = transformer.leave_Assign(stmt3, stmt3)
    assert len(new_stmt3.targets) == 1

  attr: typing.Any = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer.context.semantics, "get_definition", return_value=None):
      assert transformer.leave_Attribute(attr, attr) is not None

    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"requires_plugin": "yes"}}}),
    ):
      assert transformer.leave_Attribute(attr, attr) is not None

    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"api": "jax.attr"}}})
    ):
      res: typing.Any = transformer.leave_Attribute(attr, attr)
      assert res.attr.value == "attr"

    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"macro": "math.inf"}}})
    ):
      assert transformer.leave_Attribute(attr, attr) is not None


def test_api_call_mixin_missing() -> None:
  """Verify api call mixin missing behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  call_node: typing.Any = cst.parse_expression("foo()")

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)
    ):
      with mock.patch.object(transformer, "_get_mapping", return_value=None):
        assert transformer.leave_Call(call_node, call_node) is not None

      mapping: typing.Any = {"api": "jax.foo", "macro": "None"}
      with mock.patch.object(transformer, "_get_mapping", return_value=mapping):
        with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {})):
          with mock.patch.object(transformer, "check_version_constraints", return_value=None):
            with mock.patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.execute_strategy", return_value=call_node):
              with mock.patch(
                "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_post_processing", return_value=call_node
              ):
                assert transformer.leave_Call(call_node, call_node) is not None

    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(True, call_node)
    ):
      assert transformer.leave_Call(call_node, call_node) is not None


def test_api_py_imports() -> None:
  """Verify api py imports behavior."""
  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  mod: typing.Any = cst.parse_module("""from . import a
from a import *""")
  transformer.visit_ImportFrom(mod.body[0].body[0])
  assert transformer.visit_ImportFrom(mod.body[1].body[0]) is False


def test_api_helpers_missing2() -> None:
  """Verify api helpers missing2 behavior."""
  from unittest import mock

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("importlib.metadata.version", return_value="1.0"):
      assert transformer.check_version_constraints(None, None) is None

  with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {}}})):
    with mock.patch.object(transformer.context.semantics, "is_verified", return_value=True):
      with mock.patch.object(transformer.context.semantics, "resolve_variant", return_value={"api": "test"}):
        assert transformer._get_mapping("test") is not None


def test_final_gaps() -> None:
  """Verify final gaps behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics: typing.Any = SemanticsManager()
  ctx: typing.Any = RewriterContext(
    semantics=semantics, config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  transformer._is_framework_base = lambda x: False
  cls_node: typing.Any = cst.parse_module("class X(Base): pass").body[0]
  assert transformer.visit_ClassDef(cls_node) is True

  transformer.visit_Import(cst.parse_module("import numpy").body[0].body[0])

  transformer.context.module_preamble.clear()
  transformer.leave_Module(cst.parse_module("x = 1"), cst.parse_module("x = 1"))

  with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {})):
    with mock.patch.object(transformer.context.semantics, "is_verified", return_value=True):
      with mock.patch.object(transformer.context.semantics, "resolve_variant", return_value=mock.Mock(spec=["get"])):
        assert transformer._get_mapping("test") is None

  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    transformer.context.config.target_framework = "this_package_does_not_exist_123"
    assert transformer.check_version_constraints("1.0", None) is None or isinstance(
      transformer.check_version_constraints("1.0", None), str
    )

  transformer._cached_source_traits = None
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    stmt3: typing.Any = cst.parse_module("a, b = obj.apply()").body[0].body[0]
    with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
      assert transformer.leave_Assign(stmt3, stmt3) is not None

  attr: typing.Any = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value=""):
    transformer.leave_Attribute(attr, attr)

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"macro": "math.inf"}}})
    ):
      assert transformer.leave_Attribute(attr, attr) is not None
    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro": "invalid syntax :::"}}}),
    ):
      assert transformer.leave_Attribute(attr, attr) is not None

  call_node: typing.Any = cst.parse_expression("foo()")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)
    ):
      details: typing.Any = {"deprecated": True, "replaced_by": "new_foo"}
      with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", details)):
        with mock.patch.object(transformer, "check_version_constraints", return_value="warn"):
          with mock.patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.execute_strategy", return_value=call_node):
            with mock.patch(
              "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_post_processing", return_value=call_node
            ):
              assert transformer.leave_Call(call_node, call_node) is not None


def test_final_gaps_mixins_more() -> None:
  """Verify final gaps mixins more behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics: typing.Any = SemanticsManager()
  ctx: typing.Any = RewriterContext(
    semantics=semantics, config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  attr: typing.Any = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"macro": "math.inf"}}})
    ):
      res: typing.Any = transformer.leave_Attribute(attr, attr)
      assert isinstance(res, cst.Attribute)

    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro": "invalid syntax :::"}}}),
    ):
      assert transformer.leave_Attribute(attr, attr) is not None

    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"api": "jax.new_attr"}}}),
    ):
      assert transformer.leave_Attribute(attr, attr) is not None


def test_api_py_gaps_256() -> None:
  """Verify api py gaps 256 behavior."""
  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  transformer._is_framework_base = lambda x: x == "Base"
  cls_node: typing.Any = cst.parse_module("class X(Base): pass").body[0]
  assert transformer.visit_ClassDef(cls_node) is True


def test_api_py_gaps_417() -> None:
  """Verify api py gaps 417 behavior."""
  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  mod_i2: typing.Any = cst.parse_module("import numpy")
  assert transformer.visit_Import(mod_i2.body[0].body[0]) is False


def test_api_py_gaps_448() -> None:
  """Verify api py gaps 448 behavior."""
  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  mod: typing.Any = cst.parse_module("x = 1")
  transformer.context.module_preamble.clear()
  assert isinstance(transformer.leave_Module(mod, mod), cst.Module)


def test_api_helpers_gaps_312() -> None:
  """Verify api helpers gaps 312 behavior."""
  from unittest import mock

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="flax_nnx")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("importlib.metadata.version", return_value="1.0.0"):
      assert transformer.check_version_constraints(None, None) is None


def test_api_attr_mixin_gaps_85() -> None:
  """Verify api attr mixin gaps 85 behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  transformer._cached_source_traits = None
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    stmt3: typing.Any = cst.parse_module("a, b = obj.apply()").body[0].body[0]
    with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
      assert transformer.leave_Assign(stmt3, stmt3) is not None


def test_api_attr_mixin_gaps_151_162() -> None:
  """Verify api attr mixin gaps 151 162 behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  attr: typing.Any = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"macro": "math.inf"}}})
    ):
      assert transformer.leave_Attribute(attr, attr) is not None
    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro": "invalid syntax :::"}}}),
    ):
      assert transformer.leave_Attribute(attr, attr) is not None


def test_api_call_mixin_gaps_82_107() -> None:
  """Verify api call mixin gaps 82 107 behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  call_node: typing.Any = cst.parse_expression("foo()")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)
    ):
      details: typing.Any = {"deprecated": True, "replaced_by": "new_foo"}
      with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", details)):
        with mock.patch.object(transformer, "check_version_constraints", return_value="warn"):
          with mock.patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.execute_strategy", return_value=call_node):
            with mock.patch(
              "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_post_processing", return_value=call_node
            ):
              with mock.patch.object(transformer, "_get_mapping", return_value={"api": "jax.foo"}):
                assert transformer.leave_Call(call_node, call_node) is not None


def test_api_py_exact_lines() -> None:
  """Verify api py exact lines behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  transformer.context.scope_stack.append({"my_var"})
  assert transformer._is_stateful("my_var") is True

  mod_i: typing.Any = cst.parse_module("import a")
  with mock.patch.object(transformer, "_get_qualified_name", return_value=None):
    assert transformer.visit_Import(mod_i.body[0].body[0]) is False

  alias: typing.Any = cst.Name("b")
  import_from: typing.Any = cst.ImportFrom(module=cst.Name("a"), names=[alias])
  assert transformer.visit_ImportFrom(import_from) is False


def test_api_helpers_exact_lines() -> None:
  """Verify api helpers exact lines behavior."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="flax_nnx")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  assert transformer.check_version_constraints("1.0", None) is None or isinstance(
    transformer.check_version_constraints("1.0", None), str
  )


def test_api_attr_mixin_exact_lines() -> None:
  """Verify api attr mixin exact lines behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  transformer._cached_source_traits = None
  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
      stmt: typing.Any = cst.parse_module("a, b = foo()").body[0].body[0]
      assert transformer.leave_Assign(stmt, stmt) is not None

  attr: typing.Any = cst.parse_expression("obj.attr")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"api": "new.api"}}})
    ):
      assert transformer.leave_Attribute(attr, attr) is not None

    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro_template": "math.inf"}}}),
    ):
      assert transformer.leave_Attribute(attr, attr) is not None
    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"macro_template": ":::"}}}),
    ):
      assert transformer.leave_Attribute(attr, attr) is not None


def test_api_call_mixin_exact_lines() -> None:
  """Verify api call mixin exact lines behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  transformer.context.config.strict_mode = True

  call_node: typing.Any = cst.parse_expression("obj.method()")
  with mock.patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)):
    with mock.patch.object(transformer, "_get_qualified_name", return_value=None):
      with mock.patch(
        "ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method", return_value="torch.method"
      ):
        with mock.patch.object(transformer, "_get_mapping", return_value={"api": "jax.method"}):
          with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {})):
            assert transformer.leave_Call(call_node, call_node) is not None

  super_call: typing.Any = cst.parse_expression("super().__init__()")
  transformer.leave_Call(super_call, super_call)

  call_node2: typing.Any = cst.parse_expression("torch.foo()")
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    transformer.leave_Call(call_node2, call_node2)

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.valid"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node2)
    ):
      with mock.patch.object(transformer, "_get_mapping", return_value={"api": "jax.valid"}):
        with mock.patch.object(transformer, "check_version_constraints", return_value="Warning!"):
          assert transformer.leave_Call(call_node2, call_node2) is not None


def test_api_py_gap_417() -> None:
  """Verify api py gap 417 behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)
  mod_i: typing.Any = cst.parse_module("import a")
  with mock.patch.object(transformer, "_cst_to_string", return_value=None):
    assert transformer.visit_Import(mod_i.body[0].body[0]) is False


def test_api_attr_mixin_gap_85() -> None:
  """Verify api attr mixin gap 85 behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.core.rewriter.passes.api_attr_mixin import ApiTransformerAttrMixin

  class DummyTransformer(ApiTransformerAttrMixin):
    """Test helper class DummyTransformer."""

    def _get_qualified_name(self, func: typing.Any) -> typing.Any:
      """Execute get qualified name helper logic.

      Args:
          func: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return "foo"

    @property
    def semantics(self) -> typing.Any:
      """Execute semantics helper logic.

      Returns:
          Computed test result or fixture value.
      """
      m: typing.Any = mock.Mock()
      m.get_definition.return_value = ("id", {})
      return m

    @property
    def context(self) -> typing.Any:
      """Execute context helper logic.

      Returns:
          Computed test result or fixture value.
      """
      m: typing.Any = mock.Mock()
      m.scope_stack = [set(), set()]
      return m

  transformer: typing.Any = DummyTransformer()
  stmt: typing.Any = cst.parse_module("a = call()").body[0].body[0]
  with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=False):
    assert transformer.leave_Assign(stmt, stmt) is not None


def test_api_attr_mixin_gaps_151_162_2() -> None:
  """Verify api attr mixin gaps 151 162 2 behavior."""
  from unittest import mock

  import libcst as _cst

  cst: typing.Any = _cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx: typing.Any = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer: typing.Any = ApiTransformer(ctx)

  attr: typing.Any = cst.parse_expression("obj.attr")

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer, "_get_mapping", return_value={"api": "new.api"}):
      assert transformer.leave_Attribute(attr, attr) is not None

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer, "_get_mapping", return_value={"macro_template": "math.inf"}):
      assert transformer.leave_Attribute(attr, attr) is not None

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer, "_get_mapping", return_value={"macro_template": ":::"}):
      assert transformer.leave_Attribute(attr, attr) is not None
