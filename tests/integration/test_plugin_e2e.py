"""
End-to-End Integration Test for Plugins.

Verifies that the entire plugin pipeline works:
CLI -> Engine -> Semantics -> Hooks -> Plugin Logic -> Output.

Target Scenarios:
1. Decompose: torch.add(x, y, alpha=2) -> jax.numpy.add(x, y * 2)
2. Recompose: jax.numpy.add(x, y * 2)  -> torch.add(x, y, alpha=2)
"""

import pytest
from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.decompositions import transform_alpha_add, transform_alpha_add_reverse

# --- Mock Knowledge Base for E2E ---
# Since we removed hardcoded defaults, we must provide the knowledge
# that tells the E2E CLI to use the plugins. This simulates having valid JSONs.


class E2EMockSemantics(SemanticsManager):
  def __init__(self):
    # Skip super init to avoid file loading
    self.data = {
      "add": {
        "std_args": ["x", "y"],
        "variants": {
          "torch": {"api": "torch.add", "requires_plugin": "recompose_alpha"},
          "jax": {"api": "jax.numpy.add", "requires_plugin": "decompose_alpha"},
        },
      }
    }
    self.import_data = {}
    # Simple reverse index
    self._reverse_index = {
      "torch.add": ("add", self.data["add"]),
      "jax.numpy.add": ("add", self.data["add"]),
    }

  def get_definition(self, name):
    return self._reverse_index.get(name)

  def get_import_map(self, _target):
    return {}  # Basic for E2E

  def get_known_apis(self):
    return self.data


@pytest.fixture
def mock_cli_semantics(monkeypatch):
  """Patches the SemanticsManager used in the CLI handler."""
  # Updated path to point to 'commands' module
  monkeypatch.setattr("ml_switcheroo.cli.commands.SemanticsManager", E2EMockSemantics)


@pytest.fixture(autouse=True)
def register_test_hooks():
  """Ensure required hooks are registered in the global registry."""
  _HOOKS["decompose_alpha"] = transform_alpha_add
  _HOOKS["recompose_alpha"] = transform_alpha_add_reverse
  yield
  # No strict cleanup needed for registry in e2e context usually


def test_alpha_composition_decomposition_e2e(tmp_path, mock_cli_semantics, register_test_hooks):
  """
  Verify complete bidirectional flow using 'decompose_alpha' and 'recompose_alpha'.
  """

  # --- STEP 1: Torch -> JAX (Decomposition) ---
  infile_t2j = tmp_path / "torch_source.py"
  infile_t2j.write_text("z = torch.add(x, y, alpha=2.5)\n")
  outfile_t2j = tmp_path / "jax_output.py"

  args_t2j = ["convert", str(infile_t2j), "--out", str(outfile_t2j), "--source", "torch", "--target", "jax"]

  try:
    main(args_t2j)
  except SystemExit as e:
    assert e.code == 0

  assert outfile_t2j.exists()
  content_t2j = outfile_t2j.read_text()

  # Checks behavior of 'decompose_alpha'
  assert "import jax" in content_t2j
  assert "jax.numpy.add" in content_t2j
  assert "alpha=" not in content_t2j.replace(" ", "")  # Check ignoring space
  assert "*" in content_t2j  # Multiplication injected
  assert "2.5" in content_t2j

  # --- STEP 2: JAX -> Torch (Recomposition) ---
  # We use the valid JAX output from Step 1 as input here
  # Input: jax.numpy.add(x, y * 2.5) or x, 2.5 * y
  infile_j2t = outfile_t2j
  outfile_j2t = tmp_path / "torch_recovered.py"

  args_j2t = ["convert", str(infile_j2t), "--out", str(outfile_j2t), "--source", "jax", "--target", "torch"]

  try:
    main(args_j2t)
  except SystemExit as e:
    assert e.code == 0

  assert outfile_j2t.exists()
  content_j2t = outfile_j2t.read_text()

  # Checks behavior of 'recompose_alpha'
  assert "import torch" in content_j2t
  # Expect function swap back
  assert "torch.add" in content_j2t

  # Expect alpha to be reconstructed from the multiplication
  # Normalize whitespace to avoid failure on 'alpha = 2.5' vs 'alpha=2.5'
  normalized_j2t = content_j2t.replace(" ", "")
  assert "alpha=" in normalized_j2t
  assert "2.5" in normalized_j2t

  # Cleanup check: The plugin strip inner arithmetic?
  # Logic: argument structure should be clean
  assert "*" not in content_j2t.split("torch.add")[1]


def test_recompose_fallback(tmp_path, mock_cli_semantics):
  """
  Verify that JAX -> Torch conversion falls back to simple rename
  if the pattern (multiplication) is not present.
  """
  infile = tmp_path / "simple_jax.py"
  infile.write_text("z = jax.numpy.add(x, y)")
  outfile = tmp_path / "simple_torch.py"

  args = ["convert", str(infile), "--out", str(outfile), "--source", "jax", "--target", "torch"]

  try:
    main(args)
  except SystemExit:
    pass

  content = outfile.read_text()

  # Should simply be renamed based on the map provided by E2EMockSemantics
  assert "torch.add(x, y)" in content
  assert "alpha" not in content
