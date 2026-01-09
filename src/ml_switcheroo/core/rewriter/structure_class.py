"""
Class Structure Rewriting Logic.

Handles transformation of Class Definitions.
Logic 1: Swaps Module Base (e.g. `torch.nn.Module` -> `flax.nnx.Module`).
"""

from typing import Optional, Set, List, Any, TYPE_CHECKING
import libcst as cst

from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.escape_hatch import EscapeHatch

# Use TYPE_CHECKING to avoid circular imports only at runtime
if TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.structure import StructureStage


class ClassStructureMixin:
  """
  Mixin for transforming ClassDef nodes.

  Expects to be mixed into a class providing `semantics`, `target_fw`, and `context`.
  """

  # Cache of all known 'Module' base classes from every registered framework.
  _known_module_bases: Optional[Set[str]] = None

  def _get_target_traits(self: "StructureStage") -> StructuralTraits:
    """Retrieves structural config for the user-selected TARGET framework."""
    try:
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.target_fw)
        if config_dict and "traits" in config_dict:
          return StructuralTraits.model_validate(config_dict["traits"])
    except Exception:
      pass
    return StructuralTraits()

  def _get_target_tiers(self: "StructureStage") -> List[str]:
    """Retrieves supported tiers for the user-selected TARGET framework."""
    try:
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.target_fw)
        if config_dict and "tiers" in config_dict:
          return config_dict["tiers"]
    except Exception:
      pass
    return ["array", "neural", "extras"]

  def _lazy_load_source_bases(self: "StructureStage") -> None:
    """
    Dynamically Populates _known_module_bases from the Registry.
    """
    if self._known_module_bases is not None:
      return

    self._known_module_bases = set()

    if hasattr(self.semantics, "framework_configs"):
      for _, config in self.semantics.framework_configs.items():
        if hasattr(config, "traits") and config.traits:
          base = getattr(config.traits, "module_base", None)
          if base:
            self._known_module_bases.add(base)
        elif isinstance(config, dict):
          traits = config.get("traits")
          if traits:
            base = traits.get("module_base") if isinstance(traits, dict) else getattr(traits, "module_base", None)
            if base:
              self._known_module_bases.add(base)

  def _is_framework_base(self: "StructureStage", name: str) -> bool:
    """
    Checks if a class name corresponds to ANY known Deep Learning Module base.

    Args:
        name: The qualified name of the base class.

    Returns:
        True if it matches a known framework base.
    """
    if not name:
      return False

    self._lazy_load_source_bases()

    # 1. Exact Match via Qualified Name
    if self._known_module_bases and name in self._known_module_bases:
      return True

    # 2. Suffix Heuristic
    if self._known_module_bases:
      for known_base in self._known_module_bases:
        if known_base.endswith(f".{name}") or name.endswith(f".{known_base}"):
          return True

    return False

  def visit_ClassDef(self: "StructureStage", node: cst.ClassDef) -> Optional[bool]:
    """
    Detects if we are entering a generic ML Module class.
    """
    self._enter_scope()

    is_module = False
    for base in node.bases:
      name = self._get_qualified_name(base.value)
      if name and self._is_framework_base(name):
        is_module = True
        break

    if is_module:
      self.context.in_module_class = True

      target_tiers = self._get_target_tiers()
      if SemanticTier.NEURAL.value not in target_tiers and "neural" not in target_tiers:
        self._report_failure(f"Target framework '{self.target_fw}' does not support Neural Network class definitions.")

    return True

  def leave_ClassDef(self: "StructureStage", original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
    """
    Performs the Base Class Swap logic.
    """
    self._exit_scope()

    # Handling errors reported during visit
    if self._current_stmt_errors and self.context.in_module_class:
      target_tiers = self._get_target_tiers()
      if SemanticTier.NEURAL.value not in target_tiers and "neural" not in target_tiers:
        return EscapeHatch.mark_failure(
          original_node, f"Target framework '{self.target_fw}' does not support Neural Network class definitions."
        )

    if self.context.in_module_class:
      self.context.in_module_class = False

      traits = self._get_target_traits()
      target_base = traits.module_base

      new_bases = []
      for base in updated_node.bases:
        name = self._get_qualified_name(base.value)
        if name and self._is_framework_base(name) and target_base:
          # Create new CST Node for target base
          new_base_node = cst.Arg(value=self._create_dotted_name(target_base))
          new_bases.append(new_base_node)
        else:
          new_bases.append(base)

      updated_node = updated_node.with_changes(bases=new_bases)

    return updated_node
