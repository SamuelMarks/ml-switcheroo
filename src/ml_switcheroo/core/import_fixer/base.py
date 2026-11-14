"""Import Fixer State Management.

Defines ``ImportFixerState``, the typed shared state container coordinated by
``ImportFixer`` and its collaborators (``AttributeAliasSimplifier``,
``ImportStatementRewriter``, and ``ImportInjector``).
"""

import typing
from dataclasses import dataclass, field

import libcst as cst

from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan
from ml_switcheroo.core.scanners import get_full_name


@dataclass
class ImportFixerState:
  """Typed shared state for import rewriting, injection, and alias collapsing.

  Attributes:
      plan: Pre-calculated ``ResolutionPlan`` describing required imports and mappings.
      source_frameworks: Set of source framework root package names eligible for pruning.
      used_names: Set of identifier names referenced in the module body for dead-code elimination.
      defined_names: Set of local identifier names already bound by imports in the module.
      satisfied_injections: Set of ``ImportReq.signature`` strings already satisfied.
      path_to_alias: Mapping from dotted module paths to local alias names.
      target_framework: Optional target framework identifier from ``plan``.
  """

  plan: ResolutionPlan
  source_frameworks: set[str] = field(default_factory=set)
  used_names: set[str] = field(default_factory=set)
  defined_names: set[str] = field(default_factory=set)
  satisfied_injections: set[str] = field(default_factory=set)
  path_to_alias: dict[str, str] = field(default_factory=dict)
  target_framework: str | None = None

  @classmethod
  def create(
    cls,
    plan: ResolutionPlan,
    source_frameworks: str | typing.Iterable[str] | None = None,
    used_names: set[str] | None = None,
  ) -> "ImportFixerState":
    """Construct an initialized ``ImportFixerState`` from ``plan`` and caller options.

    Args:
        plan: Pre-calculated ``ResolutionPlan`` describing required imports and mappings.
        source_frameworks: Framework(s) to explicitly strip imports for (e.g. ``'torch'``).
        used_names: Set of names actually used in the code body.

    Returns:
        Initialized ``ImportFixerState`` instance.
    """
    if source_frameworks is None:
      normalized_sources: set[str] = set()
    elif isinstance(source_frameworks, str):
      normalized_sources = {source_frameworks}
    else:
      normalized_sources = set(source_frameworks)

    return cls(
      plan=plan,
      source_frameworks=normalized_sources,
      used_names=used_names if used_names is not None else set(),
      defined_names=set(),
      satisfied_injections=set(),
      path_to_alias=plan.path_to_alias,
      target_framework=plan.target_framework,
    )

  def track_definition(self, alias_node: cst.ImportAlias) -> None:
    """Record a local name defined by an ``ImportAlias`` node.

    If ``import torch.nn as nn``, records ``'nn'``.
    If ``import torch``, records ``'torch'``.

    Args:
        alias_node: CST ``ImportAlias`` node being retained or injected.
    """
    if alias_node.asname:
      target = alias_node.asname.name
      if isinstance(target, cst.Name):
        self.defined_names.add(target.value)
    else:
      name_val = get_full_name(alias_node.name)
      self.defined_names.add(name_val.split(".")[0])


__all__ = ["ImportFixerState"]
