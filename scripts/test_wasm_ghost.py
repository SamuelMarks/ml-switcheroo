#!/usr/bin/env python3
"""Ghost Mode (WASM) Verification Script.

This script confirms that ml-switcheroo can load its YAML/JSON framework snapshots
and perform a basic structural/semantic verification without requiring the heavy
underlying pip packages (like torch, jax) to be installed. This validates the
WebAssembly execution capability described in the paper.
"""

from pathlib import Path
import sys
from typing import List, Optional, Sequence
import unittest

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
  sys.path.insert(0, str(src_path))

from ml_switcheroo.semantics.manager import SemanticsManager  # noqa: E402

FORBIDDEN_LIBS: List[str] = ["torch", "jax", "tensorflow", "mlx"]


def check_forbidden_imports(forbidden: Optional[Sequence[str]] = None) -> bool:
  """Check whether any heavy ML libraries are imported into sys.modules.

  Args:
      forbidden: Optional sequence of forbidden library module names.

  Returns:
      True if none of the forbidden libraries are imported, False otherwise.
  """
  libs = FORBIDDEN_LIBS if forbidden is None else list(forbidden)
  for pkg in libs:
    mod = sys.modules.get(pkg)
    if mod is not None and getattr(mod, "__file__", None) is not None:
      print(f"ERROR: {pkg} is imported! Ghost mode test invalid.")
      return False
  return True


class TestWasmGhostMode(unittest.TestCase):
  """Test WASM Ghost Mode verification."""

  def test_ghost_mode_loads_snapshots(self) -> None:
    """Test that the manager can load the YAML/JSON knowledge base without ML libraries."""
    manager = SemanticsManager()

    # Test loading abstract operations (YAML)
    conv = manager.get_definition("Conv2d")
    self.assertIsNotNone(conv, "Failed to load Conv2d abstract definition.")

    # Test loading framework variants (JSON/YAML Snapshots)
    torch_variant = manager.resolve_variant("Conv2d", "torch")
    self.assertIsNotNone(torch_variant, "Failed to load Torch Conv2d variant.")
    assert torch_variant is not None
    self.assertEqual(torch_variant.get("api"), "torch.nn.Conv2d")

    flax_variant = manager.resolve_variant("Conv2d", "flax_nnx")
    self.assertIsNotNone(flax_variant, "Failed to load Flax NNX Conv2d variant.")
    assert flax_variant is not None
    self.assertEqual(flax_variant.get("api"), "flax.nnx.Conv")


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Entrypoint to run ghost mode verification.

  Args:
      argv: Optional sequence of command-line arguments.

  Returns:
      Exit code (0 on success, 1 on failure).
  """
  if not check_forbidden_imports():
    return 1

  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestWasmGhostMode)
  runner = unittest.TextTestRunner(verbosity=1)
  result = runner.run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  sys.exit(main())
