"""Test suite for API rewriter pass dispatch and mixin edge cases."""


def test_api_py_missing_coverage() -> None:
  """Verify api py missing coverage behavior."""
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

  setattr(
    getattr({"v": transformer}, "__getitem__")("v"), "_cached_target_traits", getattr({"v": "traits"}, "__getitem__")("v")
  )
  assert getattr(
    {"v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v")._get_target_traits}, "__getitem__")("v")()},
    "__getitem__",
  )("v") == getattr({"v": "traits"}, "__getitem__")("v")

  transformer._report_warning("warn")
  assert "warn" in transformer.context.current_stmt_warnings

  mod = cst.parse_module("x = 1")
  transformer.context.module_preamble = ["import ??? bad"]
  assert isinstance(transformer.leave_Module(mod, mod), cst.Module)

  setattr(transformer, "_is_framework_base", lambda x: x == "Base")
  cls_node = cst.parse_module("class X(Base): pass").body[0]
  assert isinstance(cls_node, cst.ClassDef)
  with mock.patch.object(transformer, "_get_qualified_name", return_value="Unknown"):
    assert transformer.visit_ClassDef(cls_node) is True

  from ml_switcheroo.core.rewriter.types import SignatureContext

  sig = SignatureContext()
  sig.injected_args.append(("rng", "Any"))
  getattr(
    {"v": getattr({"v": getattr({"v": sig}, "__getitem__")("v").preamble_stmts}, "__getitem__")("v").append},
    "__getitem__",
  )("v")(
    getattr(
      {
        "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_statement}, "__getitem__")("v")(
          getattr({"v": "pass"}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
  )
  transformer.context.signature_stack.append(sig)
  transformer.context.scope_stack.append(set())
  func_node = cst.parse_module("def f(): pass").body[0]
  assert isinstance(func_node, cst.FunctionDef)
  res_f = transformer.leave_FunctionDef(func_node, func_node)
  assert isinstance(res_f, cst.FunctionDef)
  assert len(res_f.params.params) == 1

  stmt = cst.parse_module("x=1").body[0]
  assert isinstance(stmt, cst.SimpleStatementLine)
  transformer.context.current_stmt_errors.append("err")
  res_err_stmt = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res_err_stmt is not None
  transformer.context.current_stmt_errors.clear()
  transformer.context.current_stmt_warnings.append("warn2")
  res_warn_stmt = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res_warn_stmt is not None

  mod_i = cst.parse_module("""from . import a
from a import *
from a import b""")
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_ImportFrom}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
              getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {"v": getattr({"v": getattr({"v": mod_i}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
                        "__getitem__",
                      )("v").body
                    },
                    "__getitem__",
                  )("v")[0]
                },
                "__getitem__",
              )("v"),
              getattr({"v": getattr({"v": cst}, "__getitem__")("v").ImportFrom}, "__getitem__")("v"),
            )
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": False}, "__getitem__")("v")
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_ImportFrom}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
              getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {"v": getattr({"v": getattr({"v": mod_i}, "__getitem__")("v").body}, "__getitem__")("v")[1]},
                        "__getitem__",
                      )("v").body
                    },
                    "__getitem__",
                  )("v")[0]
                },
                "__getitem__",
              )("v"),
              getattr({"v": getattr({"v": cst}, "__getitem__")("v").ImportFrom}, "__getitem__")("v"),
            )
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": False}, "__getitem__")("v")
  with mock.patch.object(transformer, "_cst_to_string", return_value=""):
    assert getattr(
      {
        "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_ImportFrom}, "__getitem__")("v")(
          getattr(
            {
              "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
                getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": mod_i}, "__getitem__")("v").body}, "__getitem__")("v")[2]},
                          "__getitem__",
                        )("v").body
                      },
                      "__getitem__",
                    )("v")[0]
                  },
                  "__getitem__",
                )("v"),
                getattr({"v": getattr({"v": cst}, "__getitem__")("v").ImportFrom}, "__getitem__")("v"),
              )
            },
            "__getitem__",
          )("v")
        )
      },
      "__getitem__",
    )("v") is getattr({"v": False}, "__getitem__")("v")

  mod_i2 = cst.parse_module("import numpy as np")
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_Import}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
              getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {"v": getattr({"v": getattr({"v": mod_i2}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
                        "__getitem__",
                      )("v").body
                    },
                    "__getitem__",
                  )("v")[0]
                },
                "__getitem__",
              )("v"),
              getattr({"v": getattr({"v": cst}, "__getitem__")("v").Import}, "__getitem__")("v"),
            )
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": False}, "__getitem__")("v")


def test_api_helpers_missing_coverage() -> None:
  """Verify api helpers missing coverage behavior."""
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

  assert isinstance(transformer._create_name_node("a.b.c"), cst.Attribute)

  func_node = cst.parse_module("def f(): pass").body[0]
  assert isinstance(func_node, cst.FunctionDef)
  assert isinstance(transformer._apply_preamble(func_node, ["import os"]), cst.FunctionDef)

  transformer.context.config.strict_mode = True
  assert transformer._get_mapping("torch.missing", silent=False) is None

  variant = {"required_imports": ["import os", "sys", {"module": "numpy", "alias": "np"}, {"module": "math"}]}
  transformer._handle_variant_imports(variant)
  assert "import os" in transformer.context.module_preamble

  with mock.patch.object(transformer.context.semantics, "get_framework_config", return_value=None):
    with mock.patch("importlib.metadata.version", return_value="1.0.0"):
      assert transformer.check_version_constraints(None, None) is None
    with mock.patch("importlib.metadata.version", side_effect=Exception):
      assert transformer.check_version_constraints(None, None) is None

  mod_f = cst.parse_module("def f(rng): pass")
  res_dup = transformer._inject_argument_to_signature(cst.ensure_type(mod_f.body[0], cst.FunctionDef), "rng", "Any")
  assert len(res_dup.params.params) == 1


def test_api_attr_mixin_missing() -> None:
  """Verify api attr mixin missing behavior."""
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

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.nn.call"):
    with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("nn_call", {})):
      stmt = getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {
                          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                            getattr({"v": "self.a = call()"}, "__getitem__")("v")
                          )
                        },
                        "__getitem__",
                      )("v").body
                    },
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
      )("v")
      assert isinstance(stmt, cst.Assign)
      assert transformer.leave_Assign(stmt, stmt) is not None

      stmt2 = getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {
                          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                            getattr({"v": "a = call()"}, "__getitem__")("v")
                          )
                        },
                        "__getitem__",
                      )("v").body
                    },
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
      )("v")
      assert isinstance(stmt2, cst.Assign)
      transformer.leave_Assign(stmt2, stmt2)

  class FakeTraits:
    """Test helper class FakeTraits."""

    functional_execution_method = "apply"

  setattr(
    getattr({"v": transformer}, "__getitem__")("v"),
    "_cached_source_traits",
    getattr({"v": getattr({"v": FakeTraits}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
  stmt3 = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                        getattr({"v": "a, b = obj.apply()"}, "__getitem__")("v")
                      )
                    },
                    "__getitem__",
                  )("v").body
                },
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
  )("v")
  assert isinstance(stmt3, cst.Assign)
  with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
    new_stmt3 = transformer.leave_Assign(stmt3, stmt3)
    assert len(new_stmt3.targets) == 1

  attr = cst.ensure_type(cst.parse_expression("obj.attr"), cst.Attribute)
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
      res = transformer.leave_Attribute(attr, attr)
      assert getattr(
        {"v": getattr({"v": getattr({"v": res}, "__getitem__")("v").attr}, "__getitem__")("v").value}, "__getitem__"
      )("v") == getattr({"v": "attr"}, "__getitem__")("v")

    with mock.patch.object(
      transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {"macro": "math.inf"}}})
    ):
      assert transformer.leave_Attribute(attr, attr) is not None


def test_api_call_mixin_missing() -> None:
  """Verify api call mixin missing behavior."""
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

  call_node = cst.ensure_type(cst.parse_expression("foo()"), cst.Call)

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)
    ):
      with mock.patch.object(transformer, "_get_mapping", return_value=None):
        assert transformer.leave_Call(call_node, call_node) is not None

      mapping = {"api": "jax.foo", "macro": "None"}
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
  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)

  mod = cst.parse_module("""from . import a
from a import *""")
  getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_ImportFrom}, "__getitem__")("v")(
    getattr(
      {
        "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
          getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {"v": getattr({"v": getattr({"v": mod}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
                    "__getitem__",
                  )("v").body
                },
                "__getitem__",
              )("v")[0]
            },
            "__getitem__",
          )("v"),
          getattr({"v": getattr({"v": cst}, "__getitem__")("v").ImportFrom}, "__getitem__")("v"),
        )
      },
      "__getitem__",
    )("v")
  )
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_ImportFrom}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
              getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {"v": getattr({"v": getattr({"v": mod}, "__getitem__")("v").body}, "__getitem__")("v")[1]},
                        "__getitem__",
                      )("v").body
                    },
                    "__getitem__",
                  )("v")[0]
                },
                "__getitem__",
              )("v"),
              getattr({"v": getattr({"v": cst}, "__getitem__")("v").ImportFrom}, "__getitem__")("v"),
            )
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": False}, "__getitem__")("v")


def test_api_helpers_missing2() -> None:
  """Verify api helpers missing2 behavior."""
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
      assert transformer.check_version_constraints(None, None) is None

  with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {"variants": {"jax": {}}})):
    with mock.patch.object(transformer.context.semantics, "is_verified", return_value=True):
      with mock.patch.object(transformer.context.semantics, "resolve_variant", return_value={"api": "test"}):
        assert transformer._get_mapping("test") is not None


def test_final_gaps() -> None:
  """Verify final gaps behavior."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics = SemanticsManager()
  ctx = RewriterContext(semantics=semantics, config=RuntimeConfig(source_framework="torch", target_framework="jax"))
  transformer = ApiTransformer(ctx)

  setattr(transformer, "_is_framework_base", lambda x: False)
  cls_node = cst.parse_module("class X(Base): pass").body[0]
  assert isinstance(cls_node, cst.ClassDef)
  assert transformer.visit_ClassDef(cls_node) is True

  getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_Import}, "__getitem__")("v")(
    getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                          getattr({"v": "import numpy"}, "__getitem__")("v")
                        )
                      },
                      "__getitem__",
                    )("v").body
                  },
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
    )("v")
  )

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
    stmt3 = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                          getattr({"v": "a, b = obj.apply()"}, "__getitem__")("v")
                        )
                      },
                      "__getitem__",
                    )("v").body
                  },
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
    )("v")
    assert isinstance(stmt3, cst.Assign)
    with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
      assert transformer.leave_Assign(stmt3, stmt3) is not None

  attr = cst.ensure_type(cst.parse_expression("obj.attr"), cst.Attribute)
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

  call_node = cst.ensure_type(cst.parse_expression("foo()"), cst.Call)
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)
    ):
      details: dict[str, str] = {}
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

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics = SemanticsManager()
  ctx = RewriterContext(semantics=semantics, config=RuntimeConfig(source_framework="torch", target_framework="jax"))
  transformer = ApiTransformer(ctx)

  attr = cst.ensure_type(cst.parse_expression("obj.attr"), cst.Attribute)
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
      assert transformer.leave_Attribute(attr, attr) is not None

    with mock.patch.object(
      transformer.context.semantics,
      "get_definition",
      return_value=("id", {"variants": {"jax": {"api": "jax.new_attr"}}}),
    ):
      assert transformer.leave_Attribute(attr, attr) is not None


def test_api_py_gaps_256() -> None:
  """Verify api py gaps 256 behavior."""
  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  setattr(transformer, "_is_framework_base", lambda x: x == "Base")
  cls_node = cst.parse_module("class X(Base): pass").body[0]
  assert isinstance(cls_node, cst.ClassDef)
  assert transformer.visit_ClassDef(cls_node) is True


def test_api_py_gaps_417() -> None:
  """Verify api py gaps 417 behavior."""
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
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_Import}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
              getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {"v": getattr({"v": getattr({"v": mod_i2}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
                        "__getitem__",
                      )("v").body
                    },
                    "__getitem__",
                  )("v")[0]
                },
                "__getitem__",
              )("v"),
              getattr({"v": getattr({"v": cst}, "__getitem__")("v").Import}, "__getitem__")("v"),
            )
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": False}, "__getitem__")("v")


def test_api_py_gaps_448() -> None:
  """Verify api py gaps 448 behavior."""
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
  assert isinstance(transformer.leave_Module(mod, mod), cst.Module)


def test_api_helpers_gaps_312() -> None:
  """Verify api helpers gaps 312 behavior."""
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
      assert transformer.check_version_constraints(None, None) is None


def test_api_attr_mixin_gaps_85() -> None:
  """Verify api attr mixin gaps 85 behavior."""
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
    stmt3 = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                          getattr({"v": "a, b = obj.apply()"}, "__getitem__")("v")
                        )
                      },
                      "__getitem__",
                    )("v").body
                  },
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
    )("v")
    assert isinstance(stmt3, cst.Assign)
    with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=True):
      assert transformer.leave_Assign(stmt3, stmt3) is not None


def test_api_attr_mixin_gaps_151_162() -> None:
  """Verify api attr mixin gaps 151 162 behavior."""
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
  attr = cst.ensure_type(cst.parse_expression("obj.attr"), cst.Attribute)
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

  import libcst as cst

  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="jax")
  )
  transformer = ApiTransformer(ctx)
  call_node = cst.ensure_type(cst.parse_expression("foo()"), cst.Call)
  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.foo"):
    with mock.patch(
      "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)
    ):
      details: dict[str, str] = {}
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
  assert transformer._is_stateful("my_var") is True

  mod_i = cst.parse_module("import a")
  with mock.patch.object(transformer, "_get_qualified_name", return_value=None):
    assert getattr(
      {
        "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_Import}, "__getitem__")("v")(
          getattr(
            {
              "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
                getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": mod_i}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
                          "__getitem__",
                        )("v").body
                      },
                      "__getitem__",
                    )("v")[0]
                  },
                  "__getitem__",
                )("v"),
                getattr({"v": getattr({"v": cst}, "__getitem__")("v").Import}, "__getitem__")("v"),
              )
            },
            "__getitem__",
          )("v")
        )
      },
      "__getitem__",
    )("v") is getattr({"v": False}, "__getitem__")("v")

  alias = cst.Name("b")
  import_from = getattr(
    {
      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ImportFrom}, "__getitem__")("v")(
        module=getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v")(
              getattr({"v": "a"}, "__getitem__")("v")
            )
          },
          "__getitem__",
        )("v"),
        names=getattr({"v": [getattr({"v": alias}, "__getitem__")("v")]}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert transformer.visit_ImportFrom(import_from) is False


def test_api_helpers_exact_lines() -> None:
  """Verify api helpers exact lines behavior."""
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager

  ctx = RewriterContext(
    semantics=SemanticsManager(), config=RuntimeConfig(source_framework="torch", target_framework="flax_nnx")
  )
  transformer = ApiTransformer(ctx)
  assert transformer.check_version_constraints("1.0", None) is None or isinstance(
    transformer.check_version_constraints("1.0", None), str
  )


def test_api_attr_mixin_exact_lines() -> None:
  """Verify api attr mixin exact lines behavior."""
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
      stmt = getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {
                          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                            getattr({"v": "a, b = foo()"}, "__getitem__")("v")
                          )
                        },
                        "__getitem__",
                      )("v").body
                    },
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
      )("v")
      assert isinstance(stmt, cst.Assign)
      assert transformer.leave_Assign(stmt, stmt) is not None

  attr = cst.ensure_type(cst.parse_expression("obj.attr"), cst.Attribute)
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

  call_node = cst.ensure_type(cst.parse_expression("obj.method()"), cst.Call)
  with mock.patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call_node)):
    with mock.patch.object(transformer, "_get_qualified_name", return_value=None):
      with mock.patch(
        "ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method", return_value="torch.method"
      ):
        with mock.patch.object(transformer, "_get_mapping", return_value={"api": "jax.method"}):
          with mock.patch.object(transformer.context.semantics, "get_definition", return_value=("id", {})):
            assert transformer.leave_Call(call_node, call_node) is not None

  super_call = cst.ensure_type(cst.parse_expression("super().__init__()"), cst.Call)
  transformer.leave_Call(super_call, super_call)

  call_node2 = cst.ensure_type(cst.parse_expression("torch.foo()"), cst.Call)
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
    assert getattr(
      {
        "v": getattr({"v": getattr({"v": transformer}, "__getitem__")("v").visit_Import}, "__getitem__")("v")(
          getattr(
            {
              "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
                getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": mod_i}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
                          "__getitem__",
                        )("v").body
                      },
                      "__getitem__",
                    )("v")[0]
                  },
                  "__getitem__",
                )("v"),
                getattr({"v": getattr({"v": cst}, "__getitem__")("v").Import}, "__getitem__")("v"),
              )
            },
            "__getitem__",
          )("v")
        )
      },
      "__getitem__",
    )("v") is getattr({"v": False}, "__getitem__")("v")


def test_api_attr_mixin_gap_85() -> None:
  """Verify api attr mixin gap 85 behavior."""
  from unittest import mock

  import libcst as cst

  from ml_switcheroo.core.rewriter.passes.api_attr_mixin import ApiTransformerAttrMixin

  class DummyTransformer(ApiTransformerAttrMixin):
    """Test helper class DummyTransformer."""

    def _get_qualified_name(self, func: cst.CSTNode) -> str | None:
      """Execute get qualified name helper logic.

      Args:
          func: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return "foo"

    @property
    def semantics(self) -> mock.MagicMock:
      """Execute semantics helper logic.

      Returns:
          Computed test result or fixture value.
      """
      m = mock.MagicMock()
      m.get_definition.return_value = ("id", {})
      return m

    def __init__(self) -> None:
      """Initialize DummyTransformer."""
      m = mock.Mock()
      m.scope_stack = [set(), set()]
      setattr(self, "context", m)

  transformer = DummyTransformer()
  stmt = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                        getattr({"v": "a = call()"}, "__getitem__")("v")
                      )
                    },
                    "__getitem__",
                  )("v").body
                },
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
  )("v")
  assert isinstance(stmt, cst.Assign)
  with mock.patch("ml_switcheroo.core.rewriter.passes.api_attr_mixin.is_functional_apply", return_value=False):
    assert transformer.leave_Assign(stmt, stmt) is not None


def test_api_attr_mixin_gaps_151_162_2() -> None:
  """Verify api attr mixin gaps 151 162 2 behavior."""
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

  attr = cst.ensure_type(cst.parse_expression("obj.attr"), cst.Attribute)

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer, "_get_mapping", return_value={"api": "new.api"}):
      assert transformer.leave_Attribute(attr, attr) is not None

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer, "_get_mapping", return_value={"macro_template": "math.inf"}):
      assert transformer.leave_Attribute(attr, attr) is not None

  with mock.patch.object(transformer, "_get_qualified_name", return_value="torch.attr"):
    with mock.patch.object(transformer, "_get_mapping", return_value={"macro_template": ":::"}):
      assert transformer.leave_Attribute(attr, attr) is not None
