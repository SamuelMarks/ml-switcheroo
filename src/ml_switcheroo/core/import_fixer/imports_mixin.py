"""
Import Logic Mixin.

Handles visiting, cleaning, and rewriting `Import` and `ImportFrom` nodes based
on the centralized `ResolutionPlan`.
"""

from typing import Union

import libcst as cst

from ml_switcheroo.core.import_fixer.utils import create_dotted_name
from ml_switcheroo.core.scanners import get_full_name
from ml_switcheroo.core.import_fixer.resolution import ImportReq


class ImportMixin(cst.CSTTransformer):
  """
  Mixin for processing Import statements.
  """

  def _make_alias_node(self, req: ImportReq) -> cst.ImportAlias:
    """Helper to construct CST ImportAlias from Requirement."""
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
    new_aliases = []
    replacement_occurred = False

    for alias in updated_node.names:
      full_name = get_full_name(alias.name)
      root_pkg = full_name.split(".")[0]

      # 1. Check for Specific Mapping (e.g. import torch.nn)
      if full_name in self.plan.mappings:
        req = self.plan.mappings[full_name]

        new_alias = self._make_alias_node(req)

        # Preserve alias if not specified in requirement but present in source
        if not req.alias and alias.asname and not new_alias.asname:
          new_alias = new_alias.with_changes(asname=alias.asname)

        new_aliases.append(new_alias)
        self._track_definition(new_alias)

        self._satisfied_injections.add(req.signature)
        replacement_occurred = True
        continue

      self._track_definition(alias)

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
    if not updated_node.module:
      return updated_node

    module_name = get_full_name(updated_node.module)
    root_pkg = module_name.split(".")[0]

    # Check if this statement matches a mapping key (e.g. "torch.nn")
    if len(updated_node.names) == 1 and isinstance(updated_node.names[0], cst.ImportAlias):
      import_name = updated_node.names[0].name.value
      lookup_key = f"{module_name}.{import_name}"

      if lookup_key in self.plan.mappings:
        req = self.plan.mappings[lookup_key]

        if req.subcomponent:
          # Convert to Import for robustness (prevents deep from-imports if preferred)
          new_node = cst.Import(names=[self._make_alias_node(req)])
          self._satisfied_injections.add(req.signature)
          # Track definition manually since we bypass leave_Import logic
          if isinstance(new_node.names[0], cst.ImportAlias):
            self._track_definition(new_node.names[0])
          return new_node

        else:
          new_node = cst.Import(names=[self._make_alias_node(req)])
          self._satisfied_injections.add(req.signature)
          if isinstance(new_node.names[0], cst.ImportAlias):
            self._track_definition(new_node.names[0])
          return new_node

    for alias in updated_node.names:
      if isinstance(alias, cst.ImportAlias):
        self._track_definition(alias)

    if root_pkg in self.source_fws:
      if self.preserve_source:
        return updated_node
      return cst.RemoveFromParent()

    return updated_node
