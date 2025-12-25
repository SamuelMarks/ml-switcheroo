"""
Tests for the ImportFixer AST Transformation.
Updated for Robust Scoping (Feature 026).

Verifies that:
1. Source imports are PRESERVED if aliased usages are found (robust logic).
2. 'from X import Y' bindings are tracked.
3. Plain usage tracking still works.
4. Root import switching (Case B from FromImport) works.
"""

from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.core.scanners import UsageScanner
import libcst as cst

# Define a standard map for testing to simulate SemanticsManager
TEST_MAP = {
  "torch.nn": ("flax", "linen", "nn"),
  "torch.nn.functional": ("jax", "nn", "F"),
  "torch.optim": ("optax", None, None),
  # Test Case: from Source import Sub -> import Root as Alias (sub=None)
  "flax.nnx": ("torch.nn", None, "nn"),
}


def apply_fixer(code: str, source="torch", target="jax", preserve=False) -> str:
  """Helper to parse, run ImportFixer, and emit code."""
  tree = cst.parse_module(code)
  fixer = ImportFixer(source, target, submodule_map=TEST_MAP, preserve_source=preserve)
  new_tree = tree.visit(fixer)
  return new_tree.code


def test_usage_scanner_aliased_import():
  """
  Verify scanner detects 'import torch as t' -> usage of 't'.
  """
  code = "import torch as t\nx = t.abs(y)"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)

  assert scanner.get_result() is True
  assert "t" in scanner._tracked_aliases


def test_usage_scanner_submodule_import():
  """
  Verify scanner detects 'from torch import nn' -> usage of 'nn'.
  """
  code = "from torch import nn\nlayer = nn.Linear()"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)

  assert scanner.get_result() is True
  assert "nn" in scanner._tracked_aliases


def test_usage_scanner_ignores_comments():
  """
  Verify usage inside comments doesn't trigger preservation.
  """
  code = "import torch\n# torch.abs(x) is cool"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)

  assert scanner.get_result() is False


def test_usage_scanner_ignores_strings():
  """
  Verify usage inside strings doesn't trigger preservation.
  """
  code = "import torch\nx = 'torch.abs'"
  tree = cst.parse_module(code)
  scanner = UsageScanner("torch")
  tree.visit(scanner)

  assert scanner.get_result() is False


def test_preserve_lingering_import_aliased():
  """
  Full integration check: Aliased import preserved if used.
  """
  # Scanner logic is separated in ASTEngine, but here we simulate the flag passing
  code = "import torch as t\nx = t.bad_func()"
  # preserve=True simulates scanner finding 't'
  result = apply_fixer(code, preserve=True)

  assert "import torch as t" in result


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
  # Simulate scanner finding 'torch' usage
  result = apply_fixer(code, preserve=True)

  # Check Pruning/Remapping logic
  assert "from flax import linen as nn" in result
  assert "from torch import nn" not in result

  # Check Preservation logic
  assert "import torch" in result


def test_preserve_submodule_if_used_unmapped():
  """
  Scenario: 'from torch import optim'. If 'optim' is mapped, it's replaced.
  If 'from torch import unknown_sub' is used, and scanner finds 'unknown_sub',
  ImportFixer should preserve it if preserving source enabled.
  """
  # Note: ImportFixer currently prunes 'from torch' even if preserve=True
  # UNLESS we specifically add logic to keep unmapped 'from' imports?
  # Let's check implementation.
  # Current implementation: "if from torch... return updated_node if preserve else Remove"

  code = "from torch import unknown\nx = unknown.func()"

  # Logic: Scanner finds 'unknown'. Engine passes preserve=True.
  result = apply_fixer(code, preserve=True)

  assert "from torch import unknown" in result


def test_transform_from_import_to_root_import():
  """
  Scenario: 'from flax import nnx' -> Mapped to 'torch.nn' (root='torch.nn', sub=None).
  Expect: 'import torch.nn as nnx' (or alias if specified).
  """
  code = "from flax import nnx"
  # Using the 'flax.nnx' mapping defined in TEST_MAP above which maps to (torch.nn, None, nn)

  # Run fixer with source=flax, target=torch
  tree = cst.parse_module(code)
  fixer = ImportFixer("flax", "torch", submodule_map=TEST_MAP)
  new_tree = tree.visit(fixer)
  result = new_tree.code

  # Should become 'import torch.nn as nn'
  assert "import torch.nn as nn" in result
  assert "from" not in result
