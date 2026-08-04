"""Unit tests for the BaseImportFixer configuration and state tracking.

This test suite ensures that `BaseImportFixer` behaves correctly under different
initialization scenarios (such as single framework strings, lists of frameworks,
or no frameworks) and correctly tracks imports/definitions (with or without aliases)
using LibCST node representations.
"""

import libcst as cst
from ml_switcheroo.core.import_fixer.base import BaseImportFixer
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan


def test_baseimportfixer_init():
  """Verifies the standard initialization of `BaseImportFixer`.

  Ensures that when the fixer is initialized with a ResolutionPlan, a single framework
  string, and the preserve_source flag, the internal attributes (plan, normalized
  source_fws set, preserve_source flag, empty defined_names set, and empty
  satisfied_injections set) are correctly set.

  Args:
      None

  Returns:
      None
  """
  plan = ResolutionPlan()
  fixer = BaseImportFixer(plan=plan, source_fws="torch", preserve_source=True)
  assert fixer.plan == plan
  assert fixer.source_fws == {"torch"}
  assert fixer.preserve_source is True
  assert fixer._defined_names == set()
  assert fixer._satisfied_injections == set()
  assert fixer._path_to_alias == plan.path_to_alias


def test_baseimportfixer_init_list():
  """Verifies `BaseImportFixer` initialization with a list of source frameworks.

  Ensures that when a list of multiple framework strings is passed, the internal
  `source_fws` attribute is normalized correctly into a set containing all elements.

  Args:
      None

  Returns:
      None
  """
  plan = ResolutionPlan()
  fixer = BaseImportFixer(plan=plan, source_fws=["torch", "numpy"])
  assert fixer.source_fws == {"torch", "numpy"}


def test_baseimportfixer_init_none():
  """Verifies `BaseImportFixer` initialization when no source frameworks are specified.

  Ensures that omitting `source_fws` during initialization defaults the
  `source_fws` attribute to an empty set.

  Args:
      None

  Returns:
      None
  """
  plan = ResolutionPlan()
  fixer = BaseImportFixer(plan=plan)
  assert fixer.source_fws == set()


def test_baseimportfixer_track_definition_with_alias():
  """Verifies name definition tracking for aliased imports.

  Tests that `_track_definition` correctly extracts and tracks the alias name (e.g., 'nn')
  when given a LibCST `ImportAlias` representing `import torch.nn as nn`.

  Args:
      None

  Returns:
      None
  """
  plan = ResolutionPlan()
  fixer = BaseImportFixer(plan)

  # import torch.nn as nn
  alias_node = cst.ImportAlias(
    name=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")), asname=cst.AsName(name=cst.Name("nn"))
  )
  fixer._track_definition(alias_node)
  assert "nn" in fixer._defined_names


def test_baseimportfixer_track_definition_without_alias():
  """Verifies name definition tracking for non-aliased imports.

  Tests that `_track_definition` correctly extracts and tracks the root package name
  (e.g., 'torch') when given a LibCST `ImportAlias` representing `import torch.nn`.

  Args:
      None

  Returns:
      None
  """
  plan = ResolutionPlan()
  fixer = BaseImportFixer(plan)

  # import torch.nn
  alias_node = cst.ImportAlias(name=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")))
  fixer._track_definition(alias_node)
  assert "torch" in fixer._defined_names
