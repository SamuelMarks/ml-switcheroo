"""Import Logic Mixin.

Handles visiting, cleaning, and rewriting `Import` and `ImportFrom` nodes based
on the centralized `ResolutionPlan`.
"""

from typing import Union

import libcst as cst
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan

from ml_switcheroo.core.import_fixer.utils import create_dotted_name
from ml_switcheroo.core.scanners import get_full_name
from ml_switcheroo.core.import_fixer.resolution import ImportReq


class ImportMixin(cst.CSTTransformer):
  """Mixin for transforming imports during CST traversal.

  This class provides methods to process, rewrite, and clean up `Import`
  and `ImportFrom` nodes based on a centralized resolution plan.

  Attributes:
      plan: The resolution plan containing mappings and required imports.
      source_fws: List of target source frameworks.
      preserve_source: Whether to preserve source imports if not matched.
      _track_definition: A set or callback to track imported definition names.
      _satisfied_injections: A set of signatures of satisfied injected imports.
  """

  plan: ResolutionPlan
  source_fws: "list[str]"
  preserve_source: bool
  _track_definition: "set[str]"
  _satisfied_injections: "set[str]"

  def _make_alias_node(self, req: ImportReq) -> cst.ImportAlias:
    """Helper to construct a CST ImportAlias from an import requirement.

    Handles alias redundancy, checking if the alias differs from the module
    leaf, or if a dotted path import needs an alias to bind specific names.

    Args:
        req: The import requirement details.

    Returns:
        A cst.ImportAlias node representing the target import.
    """
    name_str = f"{req.module}.{req.subcomponent}" if req.subcomponent else req.module

    asname_node = None
    should_alias = False

    # Logic for alias redundancy:
    if req.alias:
      leaf = req.subcomponent if req.subcomponent else req.module.split(".")[-1]

      # 1. Alias differs from leaf? -> Use alias.
      if req.alias != leaf:
        should_alias = True

      # 2. Dotted path import needs alias to bind specific name (flattening)?
      # e.g. import torch.nn as nn OR import flax.nnx as nnx
      # We check name_str (the full path) for dots
      if "." in name_str:
        should_alias = True

    if should_alias and req.alias:
      asname_node = cst.AsName(name=cst.Name(req.alias))

    return cst.ImportAlias(name=create_dotted_name(name_str), asname=asname_node)

  def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> Union[cst.Import, cst.RemovalSentinel]:
    """Execute rewrite/removal logic on `Import` nodes after visiting.

    Iterates through names in the import node, checking for mappings and updating
    or pruning them based on whether they belong to the source frameworks.

    Args:
        original_node: The original Import node before traversal.
        updated_node: The updated Import node.

    Returns:
        The modified Import node, or RemoveFromParent() if all names are pruned.
    """
    new_aliases = []
    replacement_occurred = False

    for alias in updated_node.names:
      full_name = get_full_name(alias.name)
      root_pkg = full_name.split(".")[0]

      # 1. Check for Specific Mapping (e.g. import torch.nn)
      if full_name in self.plan.mappings and root_pkg in self.source_fws:
        req = self.plan.mappings[full_name]

        new_alias = self._make_alias_node(req)

        # Preserve alias if not specified in requirement but present in source
        if not req.alias and alias.asname and not new_alias.asname:
          new_alias = new_alias.with_changes(asname=alias.asname)

        new_aliases.append(new_alias)
        self._track_definition(new_alias)  # type: ignore

        self._satisfied_injections.add(req.signature)
        replacement_occurred = True
        continue

      self._track_definition(alias)  # type: ignore

      # 2. Existence Check
      for req in self.plan.required_imports:
        if req.module == full_name and not req.subcomponent:
          self._satisfied_injections.add(req.signature)

      # 3. Prune
      if root_pkg in self.source_fws:
        if self.preserve_source and not replacement_occurred:
          new_aliases.append(alias)
        continue

      new_aliases.append(alias)

    if not new_aliases:
      return cst.RemoveFromParent()

    return updated_node.with_changes(names=new_aliases)

  def leave_ImportFrom(
    self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
  ) -> Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel]:
    """Execute rewrite/removal logic on `ImportFrom` nodes after visiting.

    Checks the imported module and individual names, applying mapping replacements
    and converting them to standard `Import` nodes if specified by the plan.
    Otherwise, prunes them if they are from the source frameworks and not preserved.

    Args:
        original_node: The original ImportFrom node before traversal.
        updated_node: The updated ImportFrom node.

    Returns:
        The modified node (ImportFrom or Import), or RemoveFromParent() if pruned.
    """
    if not updated_node.module:
      return updated_node

    module_name = get_full_name(updated_node.module)
    root_pkg = module_name.split(".")[0]

    if isinstance(updated_node.names, cst.ImportStar):
      if root_pkg in self.source_fws and not getattr(self, "preserve_source", False):
        return cst.RemoveFromParent()
      return updated_node

    # Check if this statement matches a mapping key (e.g. "torch.nn")
    if len(updated_node.names) == 1 and isinstance(updated_node.names[0], cst.ImportAlias):
      import_name = updated_node.names[0].name.value
      lookup_key = f"{module_name}.{import_name}"

      if lookup_key in self.plan.mappings and root_pkg in self.source_fws:
        req = self.plan.mappings[lookup_key]

        if req.subcomponent:
          # Convert to Import for robustness (prevents deep from-imports if preferred)
          new_node = cst.Import(names=[self._make_alias_node(req)])
          self._satisfied_injections.add(req.signature)
          # Track definition manually since we bypass leave_Import logic
          if isinstance(new_node.names[0], cst.ImportAlias):
            self._track_definition(new_node.names[0])  # type: ignore
          return new_node

        else:
          new_node = cst.Import(names=[self._make_alias_node(req)])
          self._satisfied_injections.add(req.signature)
          if isinstance(new_node.names[0], cst.ImportAlias):
            self._track_definition(new_node.names[0])  # type: ignore
          return new_node

    for alias in updated_node.names:
      if isinstance(alias, cst.ImportAlias):
        self._track_definition(alias)  # type: ignore

    if root_pkg in self.source_fws:
      if self.preserve_source:
        return updated_node
      return cst.RemoveFromParent()

    return updated_node
