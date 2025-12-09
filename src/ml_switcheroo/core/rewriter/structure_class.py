"""
Class Structure Rewriting Logic.

Handles transformations relative to class definitions, specifically:
1.  Inheritance Warping: Swapping base classes (e.g. `torch.nn.Module` -> `flax.nnx.Module`).
2.  Context Tracking: Setting flags for nested methods (neural vs generic).

Refactor Note:
    Logic is now fully dynamic. It queries the SemanticsManager for registered
    framework traits to identify module bases, rather than using hardcoded literals.
"""

from typing import Optional, Set
import libcst as cst

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.semantics.schema import StructuralTraits


class ClassStructureMixin(BaseRewriter):
  """
  Mixin for transforming ClassDef nodes.

  Attributes:
      _known_module_bases (Optional[Set[str]]): A cached set of fully qualified
          class names that act as Neural Module bases (e.g., 'torch.nn.Module').
          Populated based on registered adapters and JSON config.
  """

  _known_module_bases: Optional[Set[str]] = None

  def _get_traits(self) -> StructuralTraits:
    """Retrieves structural traits for the current target framework."""
    try:
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.target_fw)
        if config_dict and "traits" in config_dict:
          return StructuralTraits.model_validate(config_dict["traits"])
    except Exception:
      pass

    # No defaults. If adapter/json is missing, framework has no structural rules.
    return StructuralTraits()

  def _lazy_load_bases(self):
    """
    Populates the cache of known framework bases from the Semantics Manager.
    This ensures that new frameworks (added via adapters/JSON) are automatically detected.
    """
    if self._known_module_bases is not None:
      return

    self._known_module_bases = set()

    # Iterate over all frameworks configured in the system
    # (Checking SemanticsManager.framework_configs)
    if hasattr(self.semantics, "framework_configs"):
      for _, config in self.semantics.framework_configs.items():
        # config can be a Pydantic model or a Dict depending on loading source
        # (JSON loader uses dicts, Adapter hydration uses pydantic/dicts mix)

        # Path A: Pydantic Object
        if hasattr(config, "traits") and config.traits:
          base = getattr(config.traits, "module_base", None)
          if base:
            self._known_module_bases.add(base)

        # Path B: Dictionary
        elif isinstance(config, dict):
          traits = config.get("traits")
          if traits:
            if isinstance(traits, dict):
              base = traits.get("module_base")
            else:
              # Handle object fallback if nested object
              base = getattr(traits, "module_base", None)

            if base:
              self._known_module_bases.add(base)

  def _is_framework_base(self, name: str) -> bool:
    """
    Checks if a class name corresponds to a known Deep Learning Module base.

    Args:
        name: The qualified name of the base class (e.g., 'torch.nn.Module').

    Returns:
        bool: True if the name matches a registered framework's module_base.
    """
    if not name:
      return False

    # Ensure cache is populated
    self._lazy_load_bases()

    # 1. Exact Match (e.g. 'torch.nn.Module' matches 'torch.nn.Module')
    if name in self._known_module_bases:
      return True

    # 2. Relaxed Matching for common aliases
    # If the user has 'nn.Module' but config says 'torch.nn.Module',
    # we check if the suffix matches to handle imperfect alias resolution.
    for known_base in self._known_module_bases:
      if name == known_base or name.endswith(f".{known_base}") or known_base.endswith(f".{name}"):
        return True

      # Special check for common "nn.Module" shorthand if the root (torch/flax) is implicit
      if known_base.endswith(".Module") and name.endswith(".Module"):
        # e.g., known="flax.nnx.Module", provided="nnx.Module" -> Match
        known_parts = known_base.split(".")
        name_parts = name.split(".")
        if len(name_parts) >= 2 and len(known_parts) >= 2:
          # Matches if the submodule matches (e.g. nnx vs nnx)
          if name_parts[-2] == known_parts[-2]:
            return True

    return False

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    self._enter_scope()

    is_module = False
    for base in node.bases:
      name = self._get_qualified_name(base.value)
      if not name:
        continue

      if self._is_framework_base(name):
        is_module = True
        break

    if is_module:
      self._in_module_class = True

    return True

  def leave_ClassDef(self, _original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
    self._exit_scope()

    if self._in_module_class:
      self._in_module_class = False

      traits = self._get_traits()
      target_base = traits.module_base

      new_bases = []
      for base in updated_node.bases:
        name = self._get_qualified_name(base.value)

        # If it is a known framework base AND we have a target replacement, swap it.
        # If we have no target_base (empty traits), we preserve the original.
        if self._is_framework_base(name):
          if target_base:
            new_base_node = cst.Arg(value=self._create_dotted_name(target_base))
            new_bases.append(new_base_node)
            continue

        new_bases.append(base)

      updated_node = updated_node.with_changes(bases=new_bases)

    return updated_node
