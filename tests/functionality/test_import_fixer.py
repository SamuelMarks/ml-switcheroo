"""Test suite for the Import Fixer module."""

import typing

import libcst as cst

from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.core.import_fixer.resolution import ImportReq, ResolutionPlan


def apply_fixer(
  code: str, plan: typing.Any = None, used_names: typing.Optional[set[str]] = None, source_fws: typing.Any = None
) -> str:
  """Applies fixer."""
  if source_fws is None:
    source_fws = {"torch"}
  tree = cst.parse_module(code)
  if plan is None:
    plan = ResolutionPlan()

  if used_names is None:
    from ml_switcheroo.core.scanners import GlobalUsageScanner

    scanner = GlobalUsageScanner()
    tree.visit(scanner)
    used_names = scanner.used_names

  fixer = ImportFixer(plan=plan, source_fws=source_fws, used_names=used_names)
  new_tree: typing.Any = tree.visit(fixer)
  return typing.cast(str, new_tree.code)


def test_remap_and_preserve_mixed() -> None:
  """Verifies the behavior of remap and preserve mixed."""
  code: str = "\nimport torch\nfrom torch import nn\nx = torch.bad()\ny = nn.Linear()\n"
  mapping: dict[str, ImportReq] = {"torch.nn": ImportReq("flax", "linen", "nn")}
  plan = ResolutionPlan(mappings=mapping)
  result: str = apply_fixer(code, plan, used_names={"torch", "nn", "x", "y"})
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
