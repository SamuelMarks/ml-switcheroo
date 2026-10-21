"""Test suite for the Api Attr Mixin module."""

import typing
from ml_switcheroo.semantics.schema import StructuralTraits
from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.core.rewriter.passes.api_attr_mixin import ApiTransformerAttrMixin


class MockSemantics:
  """Test helper class MockSemantics."""

  def __init__(self, defs: dict[str, tuple[str, dict[str, JsonValue]]], origins: dict[str, str]) -> None:
    """Initializes the MockSemantics instance.

    Args:
        defs: Test argument or fixture.
        origins: Test argument or fixture.
    """
    self.defs = defs
    self._key_origins = origins

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.defs.get(name)


class MockContext:
  """Test helper class MockContext."""

  def __init__(self) -> None:
    """Initializes the MockContext instance."""
    self.scope_stack: list[set[str]] = [set(), set()]


class MockTransformer(ApiTransformerAttrMixin, cst.CSTTransformer):
  """Test helper class MockTransformer."""

  def __init__(
    self,
    semantics: MockSemantics,
    context: MockContext,
    traits: StructuralTraits | None = None,
  ) -> None:
    """Initializes the MockTransformer instance.

    Args:
        semantics: Test argument or fixture.
        context: Test argument or fixture.
        traits: Test argument or fixture.
    """
    setattr(getattr({"v": self}, "__getitem__")("v"), "semantics", getattr({"v": semantics}, "__getitem__")("v"))
    setattr(getattr({"v": self}, "__getitem__")("v"), "context", getattr({"v": context}, "__getitem__")("v"))
    setattr(getattr({"v": self}, "__getitem__")("v"), "target_framework", getattr({"v": "jax"}, "__getitem__")("v"))
    if traits:
      self.source_traits = traits
    self.marked: set[str] = set()

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

  def _mark_stateful(self, name: str) -> None:
    """Mark.

    Args:
        name: Test argument or fixture.
    """
    self.marked.add(name)

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
    """Map.

    Args:
        name: Test argument or fixture.
        silent: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    result = self.semantics.get_definition(name)
    if result is None:
      return None
    _, details = result
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr({"v": getattr({"v": details}, "__getitem__")("v").get}, "__getitem__")("v")(
                  getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v")
                )
              },
              "__getitem__",
            )("v").get
          },
          "__getitem__",
        )("v")(getattr({"v": getattr({"v": self}, "__getitem__")("v").target_framework}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def _handle_variant_imports(self, impl: typing.Mapping[str, JsonValue | typing.Hashable | typing.Sized]) -> None:
    """Imp.

    Args:
        impl: Test argument or fixture.
    """
    pass

  def _create_dotted_name(self, name: str) -> cst.Name | cst.Attribute:
    """Dot.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if "." in name:
      parts = name.split(".")
      return cst.Attribute(value=cst.Name(parts[0]), attr=cst.Name(parts[1]))
    return cst.Name(name)


def test_leave_Assign_state_tracking() -> None:
  """Verifies the behavior of leave Assign state tracking."""
  semantics = MockSemantics({"nn.Linear": ("Linear", {"variants": {}})}, {"Linear": "neural"})
  transformer = MockTransformer(semantics, MockContext())
  tree = cst.parse_module("self.layer = nn.Linear()\nx = nn.Linear()")
  _new_tree = tree.visit(transformer)
  assert "self.layer" in transformer.context.scope_stack[-2]
  assert "x" in transformer.marked


def test_leave_Assign_unwrap() -> None:
  """Verifies the behavior of leave Assign unwrap."""

  class Traits:
    """Traits."""

    functional_execution_method = "apply"

  semantics = MockSemantics({}, {})
  transformer = getattr(
    {
      "v": getattr({"v": MockTransformer}, "__getitem__")("v")(
        getattr({"v": semantics}, "__getitem__")("v"),
        getattr({"v": getattr({"v": MockContext}, "__getitem__")("v")()}, "__getitem__")("v"),
        traits=getattr({"v": getattr({"v": Traits}, "__getitem__")("v")()}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  tree = cst.parse_module("out, state = model.apply(params, x)")
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
                  )("v").targets
                },
                "__getitem__",
              )("v")[0]
            },
            "__getitem__",
          )("v").target
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "out"}, "__getitem__")("v")


def test_leave_Attribute() -> None:
  """Verifies the behavior of leave Attribute."""
  semantics = MockSemantics(
    {
      "torch.float32": ("dtype", {"variants": {"jax": {"api": "jnp.float32"}}, "op_type": "attribute"}),
      "torch.inf": ("inf", {"variants": {"jax": {"macro_template": "jnp.inf"}}, "op_type": "attribute"}),
      "torch.plugin_req": ("plug", {"variants": {"jax": {"requires_plugin": True}}, "op_type": "attribute"}),
      "torch.func": ("func", {"variants": {"jax": {}}, "op_type": "function", "std_args": ["a"]}),
    },
    {},
  )
  transformer = MockTransformer(semantics, MockContext())
  tree = cst.parse_module("torch.float32")
  new_tree = tree.visit(transformer)
  assert "float32" in new_tree.code
  tree = cst.parse_module("torch.inf")
  new_tree = tree.visit(transformer)
  assert "jnp.inf" in new_tree.code
  tree = cst.parse_module("torch.plugin_req")
  new_tree = tree.visit(transformer)
  assert "plugin_req" in new_tree.code
  tree = cst.parse_module("torch.func")
  new_tree = tree.visit(transformer)
  assert "func" in new_tree.code


def test_leave_Attribute_no_name() -> None:
  """Verifies the behavior when _get_qualified_name returns None."""
  semantics = MockSemantics({}, {})
  transformer = MockTransformer(semantics, MockContext())
  tree = cst.parse_module("a.b.c")
  new_tree = tree.visit(transformer)
  assert "c" in new_tree.code


def test_leave_Attribute_macro_exception() -> None:
  """Verifies the behavior of leave Attribute macro correctly handling an exception."""
  semantics = MockSemantics(
    {"torch.inf": ("inf", {"variants": {"jax": {"macro_template": "INVALID"}}, "op_type": "attribute"})}, {}
  )
  transformer = MockTransformer(semantics, MockContext())
  tree = cst.parse_module("torch.inf")
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.calls.transformers as trans

    m.setattr(trans, "rewrite_as_macro", lambda t, a, k: 1 / 0)
    new_tree = tree.visit(transformer)
    assert "inf" in new_tree.code
