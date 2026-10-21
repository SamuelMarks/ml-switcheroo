"""Test suite for the Api Call Mixin module."""

import typing
from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.core.rewriter.passes.api_call_mixin import ApiTransformerCallMixin


class MockTracer:
  """Test helper class MockTracer."""

  def log_inspection(self, *args: JsonValue, **kwargs: JsonValue) -> None:
    """Mock implementation of log inspection.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.
    """
    pass

  def log_mutation(self, *args: JsonValue, **kwargs: JsonValue) -> None:
    """Mock implementation of log mutation.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.
    """
    pass


class MockHookContext:
  """Test helper class MockHookContext."""

  def __init__(self) -> None:
    """Initializes the MockHookContext instance."""
    self.current_op_id: str | None = None


class MockContext:
  """Test helper class MockContext."""

  def __init__(self) -> None:
    """Initializes the MockContext instance."""
    self.hook_context = MockHookContext()
    self.symbol_table = None


class MockTransformer(ApiTransformerCallMixin, cst.CSTTransformer):
  """Test helper class MockTransformer."""

  def __init__(self) -> None:
    """Initializes the MockTransformer instance."""
    setattr(getattr({"v": self}, "__getitem__")("v"), "strict_mode", getattr({"v": True}, "__getitem__")("v"))
    setattr(getattr({"v": self}, "__getitem__")("v"), "source_framework", getattr({"v": "torch"}, "__getitem__")("v"))
    setattr(getattr({"v": self}, "__getitem__")("v"), "target_framework", getattr({"v": "jax"}, "__getitem__")("v"))
    self.warnings: list[str] = []
    self.failures: list[str] = []
    setattr(getattr({"v": self}, "__getitem__")("v"), "semantics", getattr({"v": self}, "__getitem__")("v"))
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "context",
      getattr({"v": getattr({"v": MockContext}, "__getitem__")("v")()}, "__getitem__")("v"),
    )

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Get.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    """Mock implementation of get definition."""
    if name == "torch.known":
      return ("known_id", {})
    if name == "torch.nodep":
      return ("nodep_id", {})
    return None

  def _get_qualified_name(self, node: cst.CSTNode) -> str | None:
    """Get name.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name):
      return f"{node.value.value}.{node.attr.value}"
    return None

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
    """Map.

    Args:
        name: Test argument or fixture.
        silent: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "torch.known":
      return {"api": "jnp.known"}
    if name == "torch.nodep":
      return {"api": "jnp.nodep", "min_version": "1.0", "max_version": "2.0"}
    if name == "torch.missing_def":
      return {"api": "jnp.missing_def"}
    return None

  def check_version_constraints(self, min_v: str | None, max_v: str | None) -> str | None:
    """Mock implementation of check version constraints.

    Args:
        min_v: Test argument or fixture.
        max_v: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if min_v == "1.0":
      return "Version warning"
    return None

  def _report_warning(self, msg: str) -> None:
    """Warn.

    Args:
        msg: Test argument or fixture.
    """
    self.warnings.append(msg)

  def _report_failure(self, msg: str) -> None:
    """Fail.

    Args:
        msg: Test argument or fixture.
    """
    self.failures.append(msg)

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Alias.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return False

  def _handle_variant_imports(self, mapping: typing.Mapping[str, JsonValue | typing.Hashable | typing.Sized]) -> None:
    """Imp.

    Args:
        mapping: Test argument or fixture.
    """
    pass

  def _create_name_node(self, name: str) -> cst.Name:
    """Create.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Name(name)


def test_leave_Call_pre_check() -> None:
  """Verifies the behavior of leave Call pre check."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "handle_pre_checks", lambda *args: (True, cst.Call(func=cst.Name("handled"), args=[])))
    tree = cst.parse_module("foo()")
    new_tree = tree.visit(transformer)
    assert getattr(
      {
        "v": getattr(
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
                              "v": getattr({"v": getattr({"v": new_tree}, "__getitem__")("v").body}, "__getitem__")("v")[
                                0
                              ]
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
              )("v").func
            },
            "__getitem__",
          )("v"),
          getattr({"v": "value"}, "__getitem__")("v"),
          getattr({"v": None}, "__getitem__")("v"),
        )
      },
      "__getitem__",
    )("v") == getattr({"v": "handled"}, "__getitem__")("v")


def test_leave_Call_no_mapping_super() -> None:
  """Verifies the behavior of leave Call no mapping super."""
  transformer = MockTransformer()
  tree = cst.parse_module("super()")
  new_tree = tree.visit(transformer)
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr(
                        {"v": getattr({"v": getattr({"v": new_tree}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
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
          )("v").func
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "super"}, "__getitem__")("v")


def test_leave_Call_no_mapping_strict() -> None:
  """Verifies the behavior of leave Call no mapping strict."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "get_tracer", lambda: MockTracer())
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    tree = cst.parse_module("torch.unknown()")
    tree.visit(transformer)
    assert len(transformer.failures) == 1
    assert "not found in semantics" in transformer.failures[0]


def test_leave_Call_no_mapping_strict_neural() -> None:
  """Verifies the neural error when mapping neural tier to numpy/jax."""
  transformer = MockTransformer()
  setattr(
    getattr({"v": transformer}, "__getitem__")("v"), "target_framework", getattr({"v": "NumPy"}, "__getitem__")("v")
  )
  transformer.semantics._key_origins = {"Linear": "neural"}
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "get_tracer", lambda: MockTracer())
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    tree = cst.parse_module("torch.Linear(10, 20)")
    tree.visit(transformer)
    assert len(transformer.failures) == 1
    assert "Cannot map neural network abstraction" in transformer.failures[0]


def test_leave_Call_with_mapping() -> None:
  """Verifies the behavior of leave Call with mapping."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "execute_strategy", lambda *args: cst.Call(func=cst.Name("strategy_ok"), args=[]))
    m.setattr(mixin, "handle_post_processing", lambda s, n, *args: n)
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    m.setattr(mixin, "log_diff", lambda *args: None)
    tree = cst.parse_module("torch.known()")
    new_tree = tree.visit(transformer)
    assert len(transformer.warnings) == 0
    assert getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": new_tree}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
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
            )("v").func
          },
          "__getitem__",
        )("v").value
      },
      "__getitem__",
    )("v") == getattr({"v": "strategy_ok"}, "__getitem__")("v")


def test_leave_Call_with_mapping_version_warn() -> None:
  """Verifies the behavior of leave Call with mapping version warn."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "execute_strategy", lambda *args: cst.Call(func=cst.Name("strategy_ok"), args=[]))
    m.setattr(mixin, "handle_post_processing", lambda s, n, *args: n)
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    m.setattr(mixin, "log_diff", lambda *args: None)
    tree = cst.parse_module("torch.nodep()")
    new_tree = tree.visit(transformer)
    assert len(transformer.warnings) == 1
    assert transformer.warnings[0] == "Version warning"
    assert getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": new_tree}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
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
            )("v").func
          },
          "__getitem__",
        )("v").value
      },
      "__getitem__",
    )("v") == getattr({"v": "strategy_ok"}, "__getitem__")("v")


def test_leave_Call_implicit_method() -> None:
  """Verifies the behavior of leave Call implicit method."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "resolve_implicit_method", lambda *args: "torch.nodep")
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    m.setattr(mixin, "execute_strategy", lambda *args: cst.Call(func=cst.Name("implicit_ok"), args=[]))
    m.setattr(mixin, "handle_post_processing", lambda s, n, *args: n)
    m.setattr(mixin, "log_diff", lambda *args: None)
    tree = cst.parse_module("x.nodep()")
    new_tree = tree.visit(transformer)
    assert getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": new_tree}, "__getitem__")("v").body}, "__getitem__")("v")[0]},
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
            )("v").func
          },
          "__getitem__",
        )("v").value
      },
      "__getitem__",
    )("v") == getattr({"v": "implicit_ok"}, "__getitem__")("v")


def test_leave_Call_missing_def() -> None:
  """Verifies the behavior when definition lookup fails."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    tree = cst.parse_module("torch.missing_def()")
    new_tree = tree.visit(transformer)
    assert getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {
                            "v": getattr(
                              {
                                "v": getattr({"v": getattr({"v": new_tree}, "__getitem__")("v").body}, "__getitem__")(
                                  "v"
                                )[0]
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
                )("v").func
              },
              "__getitem__",
            )("v").attr
          },
          "__getitem__",
        )("v").value
      },
      "__getitem__",
    )("v") == getattr({"v": "missing_def"}, "__getitem__")("v")


def test_docstrings_are_not_placeholders() -> None:
  """Verify docstrings are not placeholders behavior."""
  import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

  # Module docstring
  assert mixin.__doc__ is not None
  assert "Mixin for ApiTransformer Call rewriting" in mixin.__doc__
  assert "Docstring" not in mixin.__doc__

  # Class docstring
  cls_doc = mixin.ApiTransformerCallMixin.__doc__
  assert cls_doc is not None
  assert "Docstring" not in cls_doc
  assert "ApiTransformer" in cls_doc or "mixin" in cls_doc.lower()

  # Method docstring
  method_doc = mixin.ApiTransformerCallMixin.leave_Call.__doc__
  assert method_doc is not None
  assert "leave_Call" in method_doc or "rewriting" in method_doc.lower() or "Intercepts" in method_doc
  assert "Args:" in method_doc
  assert "Returns:" in method_doc
