"""
Class Structure Rewriting Logic.

Handles transformation of Class Definitions.
Logic 1: Swaps Module Base (e.g. `torch.nn.Module` -> `flax.nnx.Module`).
"""

from typing import Optional, Set, List
import libcst as cst

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.escape_hatch import EscapeHatch


class ClassStructureMixin(BaseRewriter):
  """
  Mixin for transforming ClassDef nodes (Logic 1).
  """

  # Cache of all known 'Module' base classes from every registered framework.
  # e.g. {"torch.nn.Module", "flax.nnx.Module", "keras.Model", "fastai.Learner"}
  _known_module_bases: Optional[Set[str]] = None

  def _get_target_traits(self) -> StructuralTraits:
    """Retrieves structural config for the user-selected TARGET framework."""
    try:
      # Access config populated from Target Adapter (e.g. src/ml_switcheroo/frameworks/jax.py)
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.target_fw)
        if config_dict and "traits" in config_dict:
          return StructuralTraits.model_validate(config_dict["traits"])
    except Exception:
      pass
    return StructuralTraits()

  def _get_target_tiers(self) -> List[str]:
    """Retrieves supported tiers for the user-selected TARGET framework."""
    try:
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.target_fw)
        if config_dict and "tiers" in config_dict:
          return config_dict["tiers"]
    except Exception:
      pass
    # Default to assuming full support if not specified (legacy behavior)
    return ["array", "neural", "extras"]

  def _lazy_load_source_bases(self):
    """
    Dynamically Populates _known_module_bases from the Registry.

    This ensures that adding a new framework file (e.g. `frameworks/new_lib.py`)
    automatically allows the engine to recognize 'new_lib' classes as modules
    without editing this rewriter.
    """
    if self._known_module_bases is not None:
      return

    self._known_module_bases = set()

    # Iterate over all frameworks loaded into SemanticsManager (Hub & Spoke)
    if hasattr(self.semantics, "framework_configs"):
      for _, config in self.semantics.framework_configs.items():
        # Extract 'module_base' from Pydantic models or Dicts
        if hasattr(config, "traits") and config.traits:
          base = getattr(config.traits, "module_base", None)
          if base:
            self._known_module_bases.add(base)
        elif isinstance(config, dict):
          traits = config.get("traits")
          if traits:
            # Handle nested dict structure from JSON
            base = traits.get("module_base") if isinstance(traits, dict) else getattr(traits, "module_base", None)
            if base:
              self._known_module_bases.add(base)

  def _is_framework_base(self, name: str) -> bool:
    """
    Checks if a class name corresponds to ANY known Deep Learning Module base.
    """
    if not name:
      return False

    self._lazy_load_source_bases()

    # 1. Exact Match via Qualified Name (e.g. 'torch.nn.Module')
    if name in self._known_module_bases:
      return True

    # 2. Suffix Heuristic
    # Handles cases where strict alias resolution might fail but the class is obvious.
    # e.g. known='flax.nnx.Module', name='nnx.Module' -> Match
    for known_base in self._known_module_bases:
      if known_base.endswith(f".{name}") or name.endswith(f".{known_base}"):
        return True

    return False

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """Detects if we are entering a generic ML Module class."""
    self._enter_scope()

    is_module = False
    for base in node.bases:
      # Resolve aliases (e.g. 'nn.Module' -> 'torch.nn.Module')
      name = self._get_qualified_name(base.value)

      if self._is_framework_base(name):
        is_module = True
        break

    if is_module:
      self._in_module_class = True

      # Safety Check: Does target support Neural Networks?
      target_tiers = self._get_target_tiers()
      if SemanticTier.NEURAL.value not in target_tiers and "neural" not in target_tiers:
        # We cannot report failure on visit easily in LibCST for the *current node*
        # without modifying it immediately.
        # However, marking it now will handle the wrapping later if we flag it.
        self._report_failure(f"Target framework '{self.target_fw}' does not support Neural Network class definitions.")

    return True

  def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
    """Performs the Base Class Swap logic."""
    self._exit_scope()

    # If the class definition failed validation (e.g. tier mismatch),
    # BaseRewriter.leave_SimpleStatementLine won't catch it because ClassDef isn't a SimpleStatementLine.
    # We must handle formatting failures here for Compound Statements.

    # Note: BaseRewriter._current_stmt_errors tracks mostly expressions inside simple statements.
    # But if we called _report_failure inside visit_ClassDef, it's appended there.
    if self._current_stmt_errors:
      # Check if error pertains to this class scope?
      # Actually _current_stmt_errors is reset on visit_SimpleStatementLine.
      # ClassDef contains statements.
      # If we reported failure in visit_ClassDef, it sits in _current_stmt_errors.
      # But visit_SimpleStatement line runs on children.
      # We need to act on errors found *at this level*.

      # Simpler: If we flagged this class, escape hatch it.
      # Since _report_failure adds to a list that might be cleared by children,
      # we check the specific condition again or use a dedicated flag in _scope_stack?

      # Let's re-verify target support to be safe and stateless.
      if self._in_module_class:
        target_tiers = self._get_target_tiers()
        if SemanticTier.NEURAL.value not in target_tiers and "neural" not in target_tiers:
          return EscapeHatch.mark_failure(
            original_node, f"Target framework '{self.target_fw}' does not support Neural Network class definitions."
          )

    if self._in_module_class:
      self._in_module_class = False

      # 1. Read Traits from Target Adapter (e.g. 'flax.nnx.Module')
      traits = self._get_target_traits()
      target_base = traits.module_base

      new_bases = []
      for base in updated_node.bases:
        name = self._get_qualified_name(base.value)

        # 2. Swap Logic
        if self._is_framework_base(name) and target_base:
          # Create new CST Node for target base
          new_base_node = cst.Arg(value=self._create_dotted_name(target_base))
          new_bases.append(new_base_node)
        else:
          # Preserve non-framework inheritance (Mixins, Object, etc.)
          new_bases.append(base)

      updated_node = updated_node.with_changes(bases=new_bases)

    return updated_node
