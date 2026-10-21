"""Base Import Fixer Logic.

Defines the base class for the ImportFixer, handling initialization, state tracking,
and configuration management via the ResolutionPlan.
"""

import typing

import libcst as cst

from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan
from ml_switcheroo.core.scanners import get_full_name


class BaseImportFixer(cst.CSTTransformer):
  """Base class for import manipulation.

  Manages configuration for source removal and target injection via `ResolutionPlan`.
  """

  def __init__(
    self,
    plan: ResolutionPlan,
    source_frameworks: str | typing.Iterable[str] | None = None,
    used_names: set[str] | None = None,
  ):
    """Initialize the fixer state.

    Args:
        plan: The pre-calculated ResolutionPlan describing required imports and mappings.
        source_frameworks: Framework(s) to explicitly strip imports for (e.g. 'torch').
        used_names: Set of names actually used in the code body (for dead-code elimination).

    """
    self.plan = plan
    self.used_names = used_names if used_names is not None else set()

    # Normalize source frameworks set
    if source_frameworks is None:
      self.source_frameworks = set()
    elif isinstance(source_frameworks, str):
      self.source_frameworks = {source_frameworks}
    else:
      self.source_frameworks = set(source_frameworks)

    # State Tracking
    self._defined_names: set[str] = set()

    # Tracks which required imports have been satisfied by existing/rewritten nodes
    self._satisfied_injections: set[str] = set()

    # Populate alias map from the PLAN
    self._path_to_alias = self.plan.path_to_alias

  def _track_definition(self, alias_node: cst.ImportAlias) -> None:
    """Record a name being defined by an import statement.

    If `import torch.nn as nn`, records 'nn'.
    If `import torch`, records 'torch'.

    Args:
        alias_node: The CST ImportAlias node.

    """
    if alias_node.asname:
      target = alias_node.asname.name
      self._defined_names.add(getattr(target, "value", str(target)))
    else:
      name_val = get_full_name(alias_node.name)
      self._defined_names.add(name_val.split(".")[0])
