"""
Tests for CLI Directory Recursion.

Verifies that:
1. `convert` accepts a directory input.
2. Output directory mirrors source structure.
3. Content is transformed correctly using aliases (e.g. jnp.abs).
"""

from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
import pytest


class MockSemantics(SemanticsManager):
  """
  Mock Manager for Directory tests.
  """

  def __init__(self):
    # We assume a clean state
    self.data = {}
    self._reverse_index = {}
    self.import_data = {}
    # Prepopulate the framework configs to allow alias resolution
    self.framework_configs = {
      "jax": {"alias": {"module": "jax.numpy", "name": "jnp"}},
      "torch": {"alias": {"module": "torch", "name": "torch"}},
    }
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # Define 'abs' mapping
    self.data["abs"] = {
      "std_args": ["x"],
      "variants": {
        "torch": {"api": "torch.abs"},
        "jax": {"api": "jax.numpy.abs"},
      },
    }
    self._reverse_index["torch.abs"] = ("abs", self.data["abs"])

  def get_definition(self, api_name):
    return self._reverse_index.get(api_name)

  def resolve_variant(self, abstract_id, target_fw):
    if abstract_id in self.data:
      return self.data[abstract_id]["variants"].get(target_fw)
    return None

  def is_verified(self, _id):
    return True

  def get_framework_aliases(self):
    # Explicit mock for alias map
    return {"jax": ("jax.numpy", "jnp"), "torch": ("torch", "torch")}


def test_recursive_directory_mirroring(tmp_path, capsys, monkeypatch):
  """
  Scenario: Input dir contains 'main.py' with 'torch.abs'.
  Action: Run convert command targeting 'jax'.
  Expect: Output dir contains 'main.py' with 'jnp.abs' and correct import.
  """
  # Patch the manager class used in the CLI handler
  monkeypatch.setattr("ml_switcheroo.cli.handlers.convert.SemanticsManager", MockSemantics)

  in_root = tmp_path / "src"
  in_root.mkdir()
  (in_root / "main.py").write_text("import torch\nx = torch.abs(y)")

  out_root = tmp_path / "dst"

  # Run CLI
  try:
    main(
      [
        "convert",
        str(in_root),
        "--out",
        str(out_root),
        "--source",
        "torch",
        "--target",
        "jax",
      ]
    )
  except SystemExit:
    pass

  assert (out_root / "main.py").exists()
  content = (out_root / "main.py").read_text()

  # Verify Import Injection
  assert "import jax.numpy as jnp" in content

  # Verify Alias Usage (jnp.abs instead of jax.numpy.abs)
  assert "jnp.abs(y)" in content


def test_directory_fails_without_out_arg(tmp_path, capsys):
  """
  Scenario: Calling convert on directory without --out.
  Expect: Handled error in logs (stdout via Rich Console).
  """
  in_root = tmp_path / "bad"
  in_root.mkdir()
  try:
    main(["convert", str(in_root)])
  except SystemExit as e:
    assert e.code != 0

  captured = capsys.readouterr()
  # Handled error uses log_error which prints to console (out)
  assert "Directory conversion requires --out" in captured.out


def test_empty_directory_handling(tmp_path, capsys):
  """
  Scenario: Source directory exists but has no python files.
  Expect: Warning log, no error exit code.
  """
  in_root = tmp_path / "empty"
  in_root.mkdir()
  try:
    main(["convert", str(in_root), "--out", str(tmp_path / "out")])
  except SystemExit:
    pass

  # Logs go to stdout for Rich console
  assert "No .py files found" in capsys.readouterr().out
