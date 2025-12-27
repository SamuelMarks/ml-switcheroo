"""
Import Logic Mixin.

Handles visiting, cleaning, and rewriting `Import` and `ImportFrom` nodes.
Responsible for stripping source framework imports and mapping specific submodules
to the target framework.
"""

from typing import Union

import libcst as cst

from ml_switcheroo.core.import_fixer.utils import create_dotted_name, get_signature
from ml_switcheroo.core.scanners import get_full_name


class ImportMixin(cst.CSTTransformer):
  """
  Mixin for processing Import statements.
  """

  def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> Union[cst.Import, cst.RemovalSentinel]:
    """
    Inspects ``import ...`` statements.
    Checks aliases against submodule_map or prunes matches to source_fw.
    """
    new_aliases = []
    replacement_occurred = False

    for alias in updated_node.names:
      full_name = get_full_name(alias.name)
      root_pkg = full_name.split(".")[0]

      self._track_definition(alias)

      # 1. Check if Target is already imported
      if root_pkg == self.target_fw:
        self._target_found = True
        new_aliases.append(alias)
        continue

      # 2. Check for Specific Mapping (e.g. import torch.nn)
      mapping = self.submodule_map.get(full_name)
      if mapping:
        tgt_root, tgt_sub, default_alias = mapping
        new_name_str = f"{tgt_root}.{tgt_sub}" if tgt_sub else tgt_root

        new_asname = None
        if default_alias:
          is_redundant = False
          root_package = new_name_str.split(".")[0]
          if default_alias == root_package:
            is_redundant = True

          if not is_redundant:
            new_asname = cst.AsName(name=cst.Name(default_alias))

        elif alias.asname:
          new_asname = alias.asname

        new_alias = cst.ImportAlias(name=create_dotted_name(new_name_str), asname=new_asname)
        self._track_definition(new_alias)
        new_aliases.append(new_alias)
        replacement_occurred = True
        continue

      # 3. Prune Source Roots
      if root_pkg in self.source_fws:
        if self.preserve_source and not replacement_occurred:
          new_aliases.append(alias)
        continue

      new_aliases.append(alias)

    if not new_aliases:
      return cst.RemoveFromParent()

    res_node = updated_node.with_changes(names=new_aliases)
    self._injected_code_sigs.add(get_signature(res_node))
    return res_node

  def leave_ImportFrom(
    self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
  ) -> Union[cst.ImportFrom, cst.Import, cst.RemovalSentinel]:
    """
    Inspects ``from ... import ...`` statements.
    Handles submodule remapping and root package pruning.
    """
    if not updated_node.module:
      return updated_node

    module_name = get_full_name(updated_node.module)
    root_pkg = module_name.split(".")[0]

    self._injected_code_sigs.add(get_signature(updated_node))

    for alias in updated_node.names:
      if isinstance(alias, cst.ImportAlias):
        self._track_definition(alias)

    # Check explicit submodule import (from torch import nn)
    if len(updated_node.names) == 1 and isinstance(updated_node.names[0], cst.ImportAlias):
      import_name = updated_node.names[0].name.value
      lookup_key = f"{module_name}.{import_name}"

      mapping = self.submodule_map.get(lookup_key)
      if mapping:
        tgt_root, tgt_sub, default_alias = mapping
        final_alias = None

        if default_alias:
          is_redundant = False
          if tgt_sub:
            if default_alias == tgt_sub:
              is_redundant = True
          elif tgt_sub is None:
            root_package = tgt_root.split(".")[0]
            if default_alias == root_package:
              is_redundant = True

          if not is_redundant:
            final_alias = cst.AsName(name=cst.Name(default_alias))
        elif updated_node.names[0].asname:
          final_alias = updated_node.names[0].asname

        if tgt_sub is None:
          # Converting FROM import to IMPORT (e.g. from flax import nnx -> import flax.nnx as nnx)
          new_node = cst.Import(names=[cst.ImportAlias(name=create_dotted_name(tgt_root), asname=final_alias)])
          if isinstance(new_node.names[0], cst.ImportAlias):
            self._track_definition(new_node.names[0])

          self._injected_code_sigs.add(get_signature(new_node))
          return new_node
        else:
          new_node = cst.ImportFrom(
            module=create_dotted_name(tgt_root),
            names=[cst.ImportAlias(name=cst.Name(tgt_sub), asname=final_alias)],
          )
          if isinstance(new_node.names[0], cst.ImportAlias):
            self._track_definition(new_node.names[0])
          self._injected_code_sigs.add(get_signature(new_node))
          return new_node

    if root_pkg in self.source_fws:
      if self.preserve_source:
        return updated_node
      return cst.RemoveFromParent()

    if root_pkg == self.target_fw:
      self._target_found = True

    return updated_node
