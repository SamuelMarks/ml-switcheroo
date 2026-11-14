#!/usr/bin/env python3
"""Ghost Mode (WASM) Verification Script.

This script confirms that ml-switcheroo can load its YAML/JSON framework snapshots
and perform a basic structural/semantic verification without requiring the heavy
underlying pip packages (like torch, jax) to be installed. This validates the
WebAssembly execution capability described in the paper.
"""

import sys
import typing
import unittest
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml_switcheroo.semantics.manager import SemanticsManager

FORBIDDEN_LIBS: list[str] = ["torch", "jax", "tensorflow", "mlx"]


def check_forbidden_imports(forbidden: typing.Sequence[str] | None = None) -> bool:
  """Check whether any heavy ML libraries are imported into sys.modules.

  Args:
      forbidden: Optional sequence of forbidden library module names.

  Returns:
      True if none of the forbidden libraries are imported, False otherwise.
  """
  libs = FORBIDDEN_LIBS if forbidden is None else list(forbidden)
  for pkg in libs:
    mod = sys.modules.get(pkg)
    try:
      mod_file = mod.__file__ if mod is not None else None
    except AttributeError:
      mod_file = None
    if mod is not None and mod_file is not None:
      print(f"ERROR: {pkg} is imported! Ghost mode test invalid.")
      return False
  return True


class TestWasmGhostMode(unittest.TestCase):
  """Test WASM Ghost Mode verification."""

  def test_ghost_mode_loads_snapshots(self) -> None:
    """Test that the manager can load the YAML/JSON knowledge base without ML libraries."""
    manager = SemanticsManager()
    dict_adapter: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])

    # Test loading abstract operations (YAML)
    conv = manager.get_definition("Conv2d")
    TypeAdapter(tuple[str, dict[str, JsonValue]]).validate_python(conv)

    # Test loading framework variants (JSON/YAML Snapshots)
    torch_variant = dict_adapter.validate_python(manager.resolve_variant("Conv2d", "torch"))
    TypeAdapter(typing.Literal["torch.nn.Conv2d"]).validate_python(torch_variant.get("api"))

    flax_variant = dict_adapter.validate_python(manager.resolve_variant("Conv2d", "flax_nnx"))
    TypeAdapter(typing.Literal["flax.nnx.Conv"]).validate_python(flax_variant.get("api"))


def main(argv: typing.Sequence[str] | None = None) -> int:
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
