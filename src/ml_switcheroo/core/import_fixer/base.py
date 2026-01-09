"""
Base Import Fixer Logic.

Defines the base class for the ImportFixer, handling initialization, state tracking,
and configuration management via the ResolutionPlan.
"""

from typing import Dict, List, Optional, Set, Tuple, Union

import libcst as cst

from ml_switcheroo.core.scanners import get_full_name
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan, ImportReq


class BaseImportFixer(cst.CSTTransformer):
  """
  Base class for import manipulation.

  Manages configuration for source removal and target injection via `ResolutionPlan`.
  """

  def __init__(
    self,
    plan: ResolutionPlan,
    source_fws: Optional[Union[str, Set[str]]] = None,
    preserve_source: bool = False,
    # Legacy arguments for backward compatibility during refactor transition
    target_fw: Optional[str] = None,
    submodule_map: Optional[Dict] = None,
    alias_map: Optional[Dict] = None,
  ):
    """
    Initializes the fixer state.

    Args:
        plan: The pre-calculated ResolutionPlan describing required imports and mappings.
        source_fws: Framework(s) to strip imports for (e.g. 'torch').
        preserve_source: If True, do not delete imports even if matched.
    """
    self.plan = plan
    self.preserve_source = preserve_source

    # Normalize source frameworks set
    if source_fws is None:
      self.source_fws = set()
    elif isinstance(source_fws, str):
      self.source_fws = {source_fws}
    else:
      self.source_fws = set(source_fws)

    # State Tracking
    self._defined_names: Set[str] = set()

    # Tracks which required imports have been satisfied by existing/rewritten nodes
    self._satisfied_injections: Set[str] = set()

    # Legacy attributes for mixin compatibility (AttributeMixin reads _path_to_alias)
    # We populate this from the PLAN now.
    self._path_to_alias = self.plan.path_to_alias

    # If legacy init usage detected from old tests, integrate map
    if submodule_map:
      # Map old structure (src -> (root, sub, alias)) to new ImportReq in plan.mappings
      for src, (root, sub, alias) in submodule_map.items():
        self.plan.mappings[src] = ImportReq(module=root, subcomponent=sub, alias=alias)

        # Also populate path_to_alias for attributes mixin
        full_path = f"{root}.{sub}" if sub else root
        if alias:
          self._path_to_alias[full_path] = alias

    if alias_map:
      for fw, (mod, alias) in alias_map.items():
        self._path_to_alias[mod] = alias
        # Add implicit requirement if used (simulate RESOLVER logic for legacy tests)
        req = ImportReq(module=mod, alias=alias)
        self.plan.required_imports.append(req)

    # We store target_fw mainly for cleaning re-exports in AttributeMixin if needed
    self.target_fw = target_fw or ""

  def _track_definition(self, alias_node: cst.ImportAlias) -> None:
    """
    Records a name being defined by an import statement.

    If `import torch.nn as nn`, records 'nn'.
    If `import torch`, records 'torch'.

    Args:
        alias_node: The CST ImportAlias node.
    """
    if alias_node.asname:
      target = alias_node.asname.name
      if isinstance(target, cst.AssignTarget):
        target = target.target
      if isinstance(target, cst.Name):
        self._defined_names.add(target.value)
    else:
      name_val = get_full_name(alias_node.name)
      self._defined_names.add(name_val.split(".")[0])
