"""Test suite for the Api Attr Mixin module."""

import typing

import libcst as cst
import pytest

from ml_switcheroo.core.rewriter.passes.api_attr_mixin import ApiTransformerAttrMixin


class MockSemantics:
  """Docstring."""

  def __init__(self, defs: dict[str, tuple[str, dict[str, typing.Any]]], origins: dict[str, str]) -> None:
    """Initializes the MockSemantics instance."""
    self.defs = defs
    self._key_origins = origins

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self.defs.get(name)


class MockContext:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockContext instance."""
    self.scope_stack: list[set[str]] = [set(), set()]


class MockTransformer(ApiTransformerAttrMixin, cst.CSTTransformer):
  """Docstring."""

  def __init__(self, semantics: MockSemantics, context: MockContext, traits: typing.Optional[typing.Any] = None) -> None:
    """Initializes the MockTransformer instance."""
    self.semantics = semantics
    self.context = context
    self.target_fw = "jax"
    if traits:
      self.source_traits = traits
    self.marked: set[str] = set()

  def _get_qualified_name(self, node: typing.Any) -> typing.Optional[str]:
    """Get name."""
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name):
      return f"{node.value.value}.{node.attr.value}"
    return None

  def _mark_stateful(self, name: str) -> None:
    """Mark."""
    self.marked.add(name)

  def _get_mapping(self, name: str, silent: bool = False) -> typing.Optional[dict[str, typing.Any]]:
    """Map."""
    result = self.semantics.get_definition(name)
    if result is None:
      return None
    _, details = result
    return details.get("variants", {}).get(self.target_fw)

  def _handle_variant_imports(self, impl: typing.Any) -> None:
    """Imp."""
    pass

  def _create_dotted_name(self, name: str) -> typing.Any:
    """Dot."""
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
  transformer = MockTransformer(semantics, MockContext(), traits=Traits())
  tree = cst.parse_module("out, state = model.apply(params, x)")
  new_tree: typing.Any = tree.visit(transformer)
  assert new_tree.body[0].body[0].targets[0].target.value == "out"


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
  new_tree: typing.Any = tree.visit(transformer)
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
  new_tree: typing.Any = tree.visit(transformer)
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
    new_tree: typing.Any = tree.visit(transformer)
    assert "inf" in new_tree.code
