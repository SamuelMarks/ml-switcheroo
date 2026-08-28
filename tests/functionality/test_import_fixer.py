"""Test suite for the Import Fixer module."""

from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan, ImportReq
import libcst as cst
import typing


def apply_fixer(code: str, plan: typing.Any = None, preserve: bool = False, source_fws: typing.Any = None) -> str:
  """Applies fixer."""
  if source_fws is None:
    source_fws = {"torch"}
  tree = cst.parse_module(code)
  if plan is None:
    plan = ResolutionPlan()
  fixer = ImportFixer(plan=plan, source_fws=source_fws, preserve_source=preserve)
  new_tree: typing.Any = tree.visit(fixer)
  return typing.cast(str, new_tree.code)


def test_remap_and_preserve_mixed() -> None:
  """Verifies the behavior of remap and preserve mixed."""
  code: str = "\nimport torch\nfrom torch import nn\nx = torch.bad()\ny = nn.Linear()\n"
  mapping: dict[str, ImportReq] = {"torch.nn": ImportReq("flax", "linen", "nn")}
  plan = ResolutionPlan(mappings=mapping)
  result: str = apply_fixer(code, plan, preserve=True)
  assert "import flax.linen as nn" in result or "from flax import linen as nn" in result
  assert "from torch import nn" not in result
  assert "import torch" in result


def test_transform_from_import_to_root_import() -> None:
  """Transforms from import to root import."""
  code: str = "from flax import nnx"
  req = ImportReq("torch.nn", None, "nn")
  mapping: dict[str, ImportReq] = {"flax.nnx": req}
  plan = ResolutionPlan(mappings=mapping)
  result: str = apply_fixer(code, plan=plan, source_fws={"flax"})
  assert "import torch.nn as nn" in result
  assert "from" not in result
