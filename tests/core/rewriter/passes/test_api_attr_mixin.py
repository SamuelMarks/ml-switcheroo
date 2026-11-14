"""Test suite for the Api Attr Mixin module."""

import typing

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api_attr_mixin import AttributeRewriteHandler
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


class MockSemantics(SemanticsManager):
  """Test double `MockSemantics` implementing `get_definition` for API attr mixin tests."""

  def __init__(self, defs: dict[str, tuple[str, dict[str, JsonValue]]], origins: dict[str, str]) -> None:
    """Initializes the MockSemantics instance.

    Args:
        defs: Input `defs` parameter value (`dict[str, tuple[str, dict[str, JsonValue]]]`).
        origins: Input `origins` parameter value (`dict[str, str]`).
    """
    super().__init__(load_defaults=False)
    self.defs = defs
    self._key_origins = origins

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self.defs.get(name)


class MockContext(RewriterContext):
  """Sentinel `MockContext` fixture class used in API attr mixin tests."""

  def __init__(self) -> None:
    """Initializes the MockContext instance."""
    super().__init__(
      SemanticsManager(load_defaults=False),
      RuntimeConfig(source_framework="torch", target_framework="jax"),
    )
    self.scope_stack = [set(), set()]


class MockTransformer(cst.CSTTransformer):
  """Test helper class MockTransformer composing ``AttributeRewriteHandler``."""

  def __init__(
    self,
    semantics: MockSemantics,
    context: MockContext,
    traits: StructuralTraits | None = None,
  ) -> None:
    """Initializes the MockTransformer instance.

    Args:
        semantics: SemanticsManager instance or mock providing operation mappings.
        context: Execution context passed to the pass or hook (`MockContext`).
        traits: Plugin or framework structural trait configuration (`StructuralTraits | None`).
    """
    self.semantics: SemanticsManager = semantics
    self.context: RewriterContext = context
    self.target_framework: str = "jax"
    if traits:
      self.source_traits = traits
    self.marked: set[str] = set()
    self._handler = AttributeRewriteHandler(self)

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """Delegate ``leave_Assign`` to ``AttributeRewriteHandler``.

    Args:
        original_node: Original ``cst.Assign`` node.
        updated_node: Updated ``cst.Assign`` node.

    Returns:
        Rewritten ``cst.Assign`` node.
    """
    return self._handler.rewrite_assign(original_node, updated_node)

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
    """Delegate ``leave_Attribute`` to ``AttributeRewriteHandler``.

    Args:
        original_node: Original ``cst.Attribute`` node.
        updated_node: Updated ``cst.Attribute`` node.

    Returns:
        Rewritten ``cst.BaseExpression``.
    """
    return self._handler.rewrite_attribute(original_node, updated_node)

  def _get_qualified_name(self, node: cst.CSTNode) -> str | None:
    """Get name.

    Args:
        node: CST or AST node under inspection or transformation.

    Returns:
        Resulting `str | None` from get qualified name.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name):
      return f"{node.value.value}.{node.attr.value}"
    return None

  def _mark_stateful(self, name: str) -> None:
    """Mark.

    Args:
        name: Identifier or operation name string.
    """
    self.marked.add(name)

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
    """Map.

    Args:
        name: Identifier or operation name string.
        silent: Input `silent` parameter value (`bool`).

    Returns:
        Resulting `dict[str, JsonValue] | None` from get mapping.
    """
    result = self.semantics.get_definition(name)
    if result is None:
      return None
    _, details = result
    variants = details.get("variants")
    if not isinstance(variants, dict):
      return None
    fw_variant = variants.get(self.target_framework)
    return fw_variant if isinstance(fw_variant, dict) else None

  def _handle_variant_imports(self, impl: typing.Mapping[str, RuntimeTensorOrValue]) -> None:
    """Imp.

    Args:
        impl: Input `impl` parameter value (`typing.Mapping[str, RuntimeTensorOrValue]`).
    """

  def _create_dotted_name(self, name: str) -> cst.Name | cst.Attribute:
    """Dot.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `cst.Name | cst.Attribute` from create dotted name.
    """
    if "." in name:
      parts = name.split(".")
      return cst.Attribute(value=cst.Name(parts[0]), attr=cst.Name(parts[1]))
    return cst.Name(name)


def test_leave_Assign_state_tracking() -> None:
  """Verify `tree.visit` emits `self.layer` for leave assign state tracking."""
  semantics = MockSemantics({"nn.Linear": ("Linear", {"variants": {}})}, {"Linear": "neural"})
  transformer = MockTransformer(semantics, MockContext())
  tree = cst.parse_module("self.layer = nn.Linear()\nx = nn.Linear()")
  _new_tree = tree.visit(transformer)
  assert "self.layer" in transformer.context.scope_stack[-2]
  assert "x" in transformer.marked


def test_leave_Assign_unwrap() -> None:
  """Verify `tree.visit` handles leave assign unwrap as expected."""
  semantics = MockSemantics({}, {})
  transformer = MockTransformer(
    semantics,
    MockContext(),
    traits=StructuralTraits(functional_execution_method="apply"),
  )
  tree = cst.parse_module("out, state = model.apply(params, x)")
  new_tree = tree.visit(transformer)
  stmt = cst.ensure_type(new_tree.body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt.body[0], cst.Assign)
  assert cst.ensure_type(assign.targets[0].target, cst.Name).value == "out"


def test_leave_Attribute() -> None:
  """Verify `tree.visit` emits `float32` for leave attribute."""
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
  """Verify `context` emits `inf` for leave attribute macro exception."""
  semantics = MockSemantics(
    {"torch.inf": ("inf", {"variants": {"jax": {"macro_template": "INVALID"}}, "op_type": "attribute"})}, {}
  )
  transformer = MockTransformer(semantics, MockContext())
  tree = cst.parse_module("torch.inf")
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.calls.transformers as trans

    def _bad_macro(t: str, a: list[str], k: list[str]) -> cst.BaseExpression:
      """Raise ``ValueError`` to simulate malformed macro template.

      Args:
          t: Macro template string.
          a: Positional args.
          k: Keyword args.

      Returns:
          Never returns normally.

      Raises:
          ValueError: Always raised.
      """
      raise ValueError("bad macro")

    m.setattr(trans, "rewrite_as_macro", _bad_macro)
    new_tree = tree.visit(transformer)
    assert "inf" in new_tree.code
