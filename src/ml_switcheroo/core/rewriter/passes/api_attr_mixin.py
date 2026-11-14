"""Attribute and Assignment Rewrite Collaborator for ``ApiTransformer``.

Provides ``AttributeRewriteHandler``, which handles stateful neural variable
tracking, functional-to-OOP tuple assignment unwrapping, and constant/attribute
remapping over ``RewriterHost``.
"""

import typing

import libcst as cst
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.core.rewriter.calls import transformers as _call_transformers
from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.utils.node_diff import capture_node_source

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.semantics.manager import SemanticsManager


class AttributeRewriteHost(typing.Protocol):
  """Structural protocol required by ``AttributeRewriteHandler``.

  Attributes:
      context: Active ``RewriterContext`` instance.
  """

  context: "RewriterContext"

  @property
  def semantics(self) -> "SemanticsManager":
    """Return active ``SemanticsManager``.

    Returns:
        Active ``SemanticsManager``.
    """
    ...

  @property
  def target_framework(self) -> str:
    """Return target framework identifier.

    Returns:
        Target framework name.
    """
    ...

  @property
  def source_traits(self) -> StructuralTraits:
    """Return ``StructuralTraits`` for the source framework.

    Returns:
        Source ``StructuralTraits``.
    """
    ...

  def _get_qualified_name(self, node: cst.BaseExpression) -> str | None:
    """Resolve ``node`` to a fully qualified API name.

    Args:
        node: Expression node to resolve.

    Returns:
        Qualified name string or ``None``.
    """
    ...

  def _mark_stateful(self, var_name: str) -> None:
    """Mark ``var_name`` as stateful in the current scope.

    Args:
        var_name: Variable name to mark.
    """
    ...

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
    """Resolve target variant mapping for ``name``.

    Args:
        name: Fully qualified source name.
        silent: Whether to suppress failure reporting.

    Returns:
        Target variant mapping dict or ``None``.
    """
    ...

  def _handle_variant_imports(self, variant: typing.Mapping[str, RuntimeTensorOrValue]) -> None:
    """Inject required imports declared on ``variant``.

    Args:
        variant: Target variant mapping.
    """
    ...

  def _create_dotted_name(self, name_str: str) -> cst.Name | cst.Attribute:
    """Build a CST ``Name`` or ``Attribute`` chain from ``name_str``.

    Args:
        name_str: Dotted identifier path.

    Returns:
        Constructed ``cst.Name`` or ``cst.Attribute``.
    """
    ...


class AttributeRewriteHandler:
  """Single-responsibility handler rewriting ``cst.Assign`` and ``cst.Attribute`` nodes.

  Attributes:
      host: Host ``ApiTransformer`` (or ``AttributeRewriteHost``) providing state and resolution.
  """

  def __init__(self, host: AttributeRewriteHost) -> None:
    """Initialize ``AttributeRewriteHandler`` bound to ``host``.

    Args:
        host: Host ``AttributeRewriteHost`` instance.
    """
    self.host: AttributeRewriteHost = host

  def _track_neural_stateful_targets(self, call_node: cst.Call, targets: typing.Sequence[cst.AssignTarget]) -> None:
    """Register assigned target names as stateful when ``call_node`` constructs a ``NEURAL`` tier layer.

    Args:
        call_node: Assigned ``cst.Call`` node.
        targets: Assignment targets from ``original_node.targets``.
    """
    func_name = self.host._get_qualified_name(call_node.func)
    if not func_name:
      return
    definition = self.host.semantics.get_definition(func_name)
    if not definition:
      return
    abstract_id, _ = definition
    try:
      origins = self.host.semantics._key_origins
    except AttributeError:
      origins = {}
    if origins.get(abstract_id) != SemanticTier.NEURAL.value:
      return
    for target in targets:
      target_name = self.host._get_qualified_name(target.target)
      if target_name:
        if target_name.startswith("self.") and len(self.host.context.scope_stack) > 1:
          self.host.context.scope_stack[-2].add(target_name)
        else:
          self.host._mark_stateful(target_name)

  def rewrite_assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """Track stateful neural assignments and unwrap functional ``(output, state)`` tuples.

    Args:
        original_node: Original ``cst.Assign`` node before child traversal.
        updated_node: Updated ``cst.Assign`` node after child traversal.

    Returns:
        Transformed or unchanged ``cst.Assign`` node.
    """
    if not isinstance(original_node.value, cst.Call):
      return updated_node

    self._track_neural_stateful_targets(original_node.value, original_node.targets)

    try:
      traits = self.host.source_traits
    except AttributeError:
      traits = StructuralTraits()

    if is_functional_apply(original_node.value, traits.functional_execution_method) and len(updated_node.targets) == 1:
      assign_target = updated_node.targets[0].target
      if isinstance(assign_target, (cst.Tuple, cst.List)) and len(assign_target.elements) > 0:
        primary_target = assign_target.elements[0].value
        if isinstance(primary_target, cst.BaseAssignTargetExpression):
          new_node = updated_node.with_changes(targets=[cst.AssignTarget(target=primary_target)])
          get_tracer().log_mutation(
            "Assignment Unwrapping",
            capture_node_source(original_node),
            capture_node_source(new_node),
          )
          return new_node

    return updated_node

  def rewrite_attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
    """Rewrite constant, dtype, enum, and macro-templated ``cst.Attribute`` expressions.

    Args:
        original_node: Original ``cst.Attribute`` node before child traversal.
        updated_node: Updated ``cst.Attribute`` node after child traversal.

    Returns:
        Rewritten ``cst.BaseExpression`` or unchanged ``updated_node``.
    """
    name = self.host._get_qualified_name(original_node)
    if not name:
      return updated_node

    lookup = self.host.semantics.get_definition(name)
    if lookup:
      _, details = lookup
      raw_variants = details.get("variants")
      target_var = raw_variants.get(self.host.target_framework) if isinstance(raw_variants, dict) else None

      if isinstance(target_var, dict) and "requires_plugin" in target_var:
        return updated_node

      op_type = details.get("op_type", "function")
      if op_type == "function" and "std_args" in details and details["std_args"]:
        return updated_node

    target_impl = self.host._get_mapping(name, silent=True)

    if target_impl:
      if "api" in target_impl and isinstance(target_impl["api"], str):
        self.host._handle_variant_imports(target_impl)
        dotted_node: cst.BaseExpression = self.host._create_dotted_name(target_impl["api"])
        return dotted_node

      if "macro_template" in target_impl and isinstance(target_impl["macro_template"], str):
        try:
          return _call_transformers.rewrite_as_macro(target_impl["macro_template"], [], [])
        except (ValueError, TypeError, KeyError, cst.ParserSyntaxError):
          pass

    return updated_node


__all__ = ["AttributeRewriteHandler", "AttributeRewriteHost"]
