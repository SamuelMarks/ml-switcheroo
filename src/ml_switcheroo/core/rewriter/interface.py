"""Interface definitions for Rewriter Passes and RewriterHost Protocol.

This module defines the abstract base class ``RewriterPass`` that all transformation
passes must implement to be compatible with ``RewriterPipeline``, as well as the
``RewriterHost`` structural protocol used across call/attribute mixins and strategies.
"""

from abc import ABC, abstractmethod
import typing

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits

if typing.TYPE_CHECKING:

  class RewriterHost(typing.Protocol):
    """Structural protocol representing the host CST rewriter (e.g., ApiTransformer).

    Provides a strongly-typed contract for call/attribute rewriting helpers,
    pre/post checks, dispatch rules, and strict guards without circular imports
    or type-ignore suppressions.
    """

    context: RewriterContext

    @property
    def semantics(self) -> SemanticsManager:
      """Return the semantics manager bound to the rewriter context.

      Returns:
          SemanticsManager instance.
      """
      ...

    @property
    def config(self) -> RuntimeConfig:
      """Return the runtime configuration bound to the rewriter context.

      Returns:
          RuntimeConfig instance.
      """
      ...

    @property
    def source_fw(self) -> str:
      """Return the source framework identifier string.

      Returns:
          Source framework key (e.g., 'torch').
      """
      ...

    @property
    def target_fw(self) -> str:
      """Return the target framework identifier string.

      Returns:
          Target framework key (e.g., 'jax').
      """
      ...

    @property
    def strict_mode(self) -> bool:
      """Return whether strict mode is enabled.

      Returns:
          True if strict mode is active, False otherwise.
      """
      ...

    @property
    def source_traits(self) -> StructuralTraits:
      """Return structural traits for the source framework.

      Returns:
          StructuralTraits instance for the source framework.
      """
      ...

    def _get_target_traits(self) -> StructuralTraits:
      """Return structural traits for the target framework.

      Returns:
          StructuralTraits instance for the target framework.
      """
      ...

    def _get_source_lifecycle_lists(self) -> typing.Tuple[typing.Set[str], typing.Set[str]]:
      """Return lifecycle strip and warning method sets for the source framework.

      Returns:
          Tuple of (strip_methods_set, warn_methods_set).
      """
      ...

    def _get_qualified_name(self, node: cst.BaseExpression) -> typing.Optional[str]:
      """Resolve a CST expression node to its fully qualified name string.

      Args:
          node: The CST expression node to resolve.

      Returns:
          Fully qualified name string if resolved, else None.
      """
      ...

    def _get_mapping(self, name: str, silent: bool = False) -> typing.Optional[typing.Dict[str, JsonValue]]:
      """Retrieve the target framework mapping for a qualified API name.

      Args:
          name: The fully qualified API name.
          silent: Whether to suppress failure reporting on missing mappings.

      Returns:
          Target variant mapping dictionary if found, else None.
      """
      ...

    def _report_failure(self, reason: str) -> None:
      """Record a failure message in the current statement error buffer.

      Args:
          reason: Failure description string.
      """
      ...

    def _report_warning(self, reason: str) -> None:
      """Record a warning message in the current statement warning buffer.

      Args:
          reason: Warning description string.
      """
      ...

    def check_version_constraints(self, min_v: typing.Optional[str], max_v: typing.Optional[str]) -> typing.Optional[str]:
      """Validate target framework version constraints for an operation mapping.

      Args:
          min_v: Optional minimum version string.
          max_v: Optional maximum version string.

      Returns:
          Warning string if version constraints are violated, else None.
      """
      ...

    def _handle_variant_imports(self, variant: typing.Dict[str, JsonValue]) -> None:
      """Inject required imports specified by a target variant mapping.

      Args:
          variant: Target variant dictionary containing 'required_imports'.
      """
      ...

    def _create_name_node(self, api_path: str) -> typing.Union[cst.Name, cst.Attribute]:
      """Construct a CST Name or Attribute chain from a dotted API path.

      Args:
          api_path: Dotted API path string.

      Returns:
          Constructed cst.Name or cst.Attribute node.
      """
      ...

    def _create_dotted_name(self, name_str: str) -> typing.Union[cst.Name, cst.Attribute]:
      """Construct a CST Name or Attribute chain from a dotted name string.

      Args:
          name_str: Dotted name string.

      Returns:
          Constructed cst.Name or cst.Attribute node.
      """
      ...

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Check whether a CST node refers to a known module alias.

      Args:
          node: The CST node to inspect.

      Returns:
          True if the node is a module alias, False otherwise.
      """
      ...

    def _mark_stateful(self, var_name: str) -> None:
      """Mark a variable name as stateful in the current scope.

      Args:
          var_name: Variable identifier to mark as stateful.
      """
      ...

    def _is_stateful(self, var_name: str) -> bool:
      """Check whether a variable name is marked stateful in any active scope.

      Args:
          var_name: Variable identifier to check.

      Returns:
          True if the variable is stateful, False otherwise.
      """
      ...


class RewriterPass(ABC):
  """Abstract contract for a transformation pass in the rewriting pipeline.

  Passes encapsulate discrete transformation logic (e.g. Structural Rewriting,
  API Remapping) and are executed sequentially by the pipeline.
  """

  @abstractmethod
  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Execute the transformation logic on the given CST module.

    Args:
        module: The input LibCST module.
        context: The shared rewriter context containing configuration and state.

    Returns:
        The transformed LibCST module.

    """
    pass
