"""Test suite for the Import Fixer module."""

from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan, ImportReq
import libcst as cst


def apply_fixer(code: str, plan=None, preserve=False, source_fws={"torch"}) -> str:
  """Applies fixer."""
  tree = cst.parse_module(code)
  if plan is None:
    plan = ResolutionPlan()
  fixer = ImportFixer(plan=plan, source_fws=source_fws, preserve_source=preserve)
  new_tree = tree.visit(fixer)
  return new_tree.code


def test_remap_and_preserve_mixed():
  """Verifies the behavior of remap and preserve mixed."""
  code = "\nimport torch\nfrom torch import nn\nx = torch.bad()\ny = nn.Linear()\n"
  mapping = {"torch.nn": ImportReq("flax", "linen", "nn")}
  plan = ResolutionPlan(mappings=mapping)
  result = apply_fixer(code, plan, preserve=True)
  assert "import flax.linen as nn" in result or "from flax import linen as nn" in result
  assert "from torch import nn" not in result
  assert "import torch" in result


def test_transform_from_import_to_root_import():
  """Transforms from import to root import."""
  code = "from flax import nnx"
  req = ImportReq("torch.nn", None, "nn")
  mapping = {"flax.nnx": req}
  plan = ResolutionPlan(mappings=mapping)
  result = apply_fixer(code, plan=plan, source_fws={"flax"})
  assert "import torch.nn as nn" in result
  assert "from" not in result
