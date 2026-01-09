"""
Tests for the ImportFixer AST Transformation.
Updated for Centralized Resolution Logic.
"""

from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan, ImportReq
from ml_switcheroo.core.scanners import UsageScanner
import libcst as cst


def apply_fixer(code: str, plan=None, preserve=False, source_fws={"torch"}) -> str:
  """Helper to parse, run ImportFixer, and emit code."""
  tree = cst.parse_module(code)

  if plan is None:
    plan = ResolutionPlan()

  fixer = ImportFixer(plan=plan, source_fws=source_fws, preserve_source=preserve)
  new_tree = tree.visit(fixer)
  return new_tree.code


def test_remap_and_preserve_mixed():
  """
  Scenario: 'import torch' and 'from torch import nn'.
  'nn' is mapped (removed/replaced).
  'torch' is used elsewhere (lingering).
  Expect: 'import torch' preserved, 'from torch import nn' replaced.
  """
  code = """ 
import torch
from torch import nn
x = torch.bad() 
y = nn.Linear() 
"""
  # Plan: Map 'torch.nn' -> flax.linen
  mapping = {"torch.nn": ImportReq("flax", "linen", "nn")}
  plan = ResolutionPlan(mappings=mapping)

  # preserve=True simulates existing usage of 'torch'
  result = apply_fixer(code, plan, preserve=True)

  # Check Remapping
  assert "import flax.linen as nn" in result or "from flax import linen as nn" in result
  assert "from torch import nn" not in result

  # Check Preservation
  assert "import torch" in result


def test_transform_from_import_to_root_import():
  """
  Scenario: 'from flax import nnx' -> Mapped to 'torch.nn' (root='torch.nn', sub=None).
  Expect: 'import torch.nn as nn' (or alias if specified).
  """
  code = "from flax import nnx"

  # Plan: Map 'flax.nnx' -> 'torch.nn'
  req = ImportReq("torch.nn", None, "nn")
  mapping = {"flax.nnx": req}
  plan = ResolutionPlan(mappings=mapping)

  # Source: flax
  result = apply_fixer(code, plan=plan, source_fws={"flax"})

  assert "import torch.nn as nn" in result
  assert "from" not in result
