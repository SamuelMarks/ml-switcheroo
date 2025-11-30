"""
Tests for Recursive Directory Processing in CLI.

Verifies that:
1. Directory inputs trigger recursion.
2. Structure is mirrored in output.
3. Nested files are processed.
4. Non-Python files are ignored (implicitly or explicitly).
"""

from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.semantics.manager import SemanticsManager


# Helper to mock SemanticsManager globally effectively for the CLI test
class MockSemantics(SemanticsManager):
  def get_definition(self, api_name):
    if api_name == "torch.abs":
      return "abs", {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}}
    elif api_name == "torch.sum":
      return "sum", {"std_args": ["x"], "variants": {"torch": {"api": "torch.sum"}, "jax": {"api": "jax.numpy.sum"}}}
    else:
      return None  # Default behavior


def test_recursive_directory_mirroring(tmp_path, capsys, monkeypatch):
  """
  Scenario: Input is a directory with nested structure.
  Expectation: Output directory mirrors structure and files are converted.
  """
  # 0. Setup Mock Semantics Manager
  # We patch the SemanticsManager inside commands because main delegates to it
  monkeypatch.setattr("ml_switcheroo.cli.commands.SemanticsManager", MockSemantics)

  # 1. Setup Input Structure
  in_root = tmp_path / "src_project"
  in_root.mkdir()

  (in_root / "main.py").write_text("import torch\nx = torch.abs(y)")

  sub_dir = in_root / "utils"
  sub_dir.mkdir()
  (sub_dir / "helper.py").write_text("import torch\nz = torch.sum(a)")

  # Non-python file (should be ignored)
  (in_root / "README.md").write_text("# Docs")

  # 2. Setup Output
  out_root = tmp_path / "dst_project"  # Should be created

  # 3. Run CLI
  args = ["convert", str(in_root), "--out", str(out_root)]

  try:
    main(args)
  except SystemExit as e:
    # 0 is success
    assert e.code == 0

  # 4. Verify Output Structure
  assert out_root.exists()
  assert (out_root / "main.py").exists()
  assert (out_root / "utils" / "helper.py").exists()

  # 5. Verify Content Transformation
  main_content = (out_root / "main.py").read_text()
  assert "jax.numpy.abs" in main_content

  helper_content = (out_root / "utils" / "helper.py").read_text()
  assert "jax.numpy.sum" in helper_content

  # ignored file check
  assert not (out_root / "README.md").exists()

  # 6. Verify Rich Table Output (Feature 038)
  captured = capsys.readouterr()
  assert "Transpilation Report" not in captured.out
  # Why? Because in test_recursive_directory_mirroring, everything passes perfectly.
  # The CLI summary logic says "if failures == 0: ... success message".
  assert "Batch Complete" in captured.out
  assert "2/2 files converted perfectly" in captured.out


def test_directory_fails_without_out_arg(tmp_path, capsys):
  """
  Scenario: Directory input but no --out arg.
  Expectation: Error message and exit(1).
  """
  in_root = tmp_path / "bad_invoke"
  in_root.mkdir()

  args = ["convert", str(in_root)]

  try:
    main(args)
  except SystemExit as e:
    assert e.code == 1

  captured = capsys.readouterr()
  assert "requires --out destination" in captured.out


def test_empty_directory_handling(tmp_path, capsys):
  """
  Scenario: Directory has no py files.
  Expectation: Warning message, exit 0 (success but no work).
  """
  in_root = tmp_path / "empty_dir"
  in_root.mkdir()
  out_root = tmp_path / "out_empty"

  args = ["convert", str(in_root), "--out", str(out_root)]

  try:
    main(args)
  except SystemExit as e:
    assert e.code == 0

  captured = capsys.readouterr()
  assert "No .py files found" in captured.out
