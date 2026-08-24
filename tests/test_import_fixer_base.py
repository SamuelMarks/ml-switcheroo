"""Test module."""

import libcst as cst
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan
from ml_switcheroo.core.import_fixer.base import BaseImportFixer


def test_base_import_fixer_init():
  """Test element."""
  plan = ResolutionPlan()

  # default init
  fixer = BaseImportFixer(plan)
  assert fixer.source_fws == set()
  assert fixer.preserve_source is False
  assert fixer._defined_names == set()
  assert fixer._satisfied_injections == set()
  assert fixer._path_to_alias == {}

  # string source_fws
  fixer2 = BaseImportFixer(plan, source_fws="torch")
  assert fixer2.source_fws == {"torch"}

  # set source_fws
  fixer3 = BaseImportFixer(plan, source_fws={"torch", "keras"})
  assert fixer3.source_fws == {"torch", "keras"}


def test_track_definition():
  """Test element."""
  plan = ResolutionPlan()
  fixer = BaseImportFixer(plan)

  # import torch.nn as nn
  alias_node = cst.ImportAlias(
    name=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")), asname=cst.AsName(name=cst.Name("nn"))
  )
  fixer._track_definition(alias_node)
  assert "nn" in fixer._defined_names

  # import torch
  alias_node2 = cst.ImportAlias(name=cst.Name("torch"))
  fixer._track_definition(alias_node2)
  assert "torch" in fixer._defined_names

  # import torch.nn.functional
  alias_node3 = cst.ImportAlias(
    name=cst.Attribute(value=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")), attr=cst.Name("functional"))
  )
  fixer._track_definition(alias_node3)
  assert "torch" in fixer._defined_names
