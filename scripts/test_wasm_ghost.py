#!/usr/bin/env python3
"""Ghost Mode (WASM) Verification Script.

This script confirms that ml-switcheroo can load its YAML/JSON framework snapshots
and perform a basic structural/semantic verification without requiring the heavy
underlying pip packages (like torch, jax) to be installed. This validates the
WebAssembly execution capability described in the paper.
"""

import sys
import unittest

from ml_switcheroo.semantics.manager import SemanticsManager

# Try to ensure heavy ML libs are NOT imported
for forbidden in ["torch", "jax", "tensorflow", "mlx"]:
  if forbidden in sys.modules:
    print(f"ERROR: {forbidden} is imported! Ghost mode test invalid.")
    sys.exit(1)


class TestWasmGhostMode(unittest.TestCase):
  """Test WASM Ghost Mode."""

  def test_ghost_mode_loads_snapshots(self):
    """Test that the manager can load the YAML/JSON knowledge base without ML libraries."""
    manager = SemanticsManager()

    # Test loading abstract operations (YAML)
    conv = manager.get_definition("Conv2d")
    self.assertIsNotNone(conv, "Failed to load Conv2d abstract definition.")

    # Test loading framework variants (JSON/YAML Snapshots)
    torch_variant = manager.resolve_variant("Conv2d", "torch")
    self.assertIsNotNone(torch_variant, "Failed to load Torch Conv2d variant.")
    self.assertEqual(torch_variant.get("api"), "torch.nn.Conv2d")

    jax_variant = manager.resolve_variant("Conv2d", "jax")
    self.assertIsNotNone(jax_variant, "Failed to load JAX Conv2d variant.")


if __name__ == "__main__":
  unittest.main()
