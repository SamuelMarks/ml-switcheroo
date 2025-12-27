"""
Base Import Fixer Logic.

Defines the base class for the ImportFixer, handling initialization, state tracking,
and configuration management.
"""

from typing import Dict, List, Optional, Set, Tuple, Union

import libcst as cst

from ml_switcheroo.core.scanners import get_full_name


class BaseImportFixer(cst.CSTTransformer):
  """
  Base class for import manipulation.

  Manages configuration for source removal and target injection, and tracks
  the state of the module being transforming (e.g., defined aliases, existing imports).
  """

  def __init__(
    self,
    source_fws: Union[str, List[str]],
    target_fw: str,
    submodule_map: Dict[str, Tuple[str, Optional[str], Optional[str]]],
    alias_map: Optional[Dict[str, Tuple[str, str]]] = None,
    preserve_source: bool = False,
  ):
    """
    Initializes the fixer state.

    Args:
        source_fws: Framework(s) to strip imports for (e.g. 'torch' or ['torch', 'flax']).
        target_fw: The target framework we are converting to.
        submodule_map: Mapping for rewriting specific imports (source_path -> target_config).
        alias_map: Configuration for standard aliases (e.g. jax -> jnp).
        preserve_source: If True, do not delete imports even if matched.
    """
    if isinstance(source_fws, str):
      self.source_fws = {source_fws}
    else:
      self.source_fws = set(source_fws)

    self.target_fw = target_fw
    self.submodule_map = submodule_map
    self.alias_map = alias_map or {}
    self.preserve_source = preserve_source

    # State Tracking
    self._target_found = False
    self._defined_names: Set[str] = set()
    self._injected_code_sigs: Set[str] = set()

    # Build Collapsing Map: target_full_path -> alias
    # Used for collapsing `jax.numpy.abs` -> `jnp.abs`
    self._path_to_alias: Dict[str, str] = {}

    # 1. From Submodules Config (Result of Tier Mapping)
    for _, (root, sub, alias) in self.submodule_map.items():
      if alias:
        full_path = f"{root}.{sub}" if sub else root
        self._path_to_alias[full_path] = alias

    # 2. From Framework Aliases Config
    for _, (mod, alias) in self.alias_map.items():
      if alias:
        self._path_to_alias[mod] = alias

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
