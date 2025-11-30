"""
Tests for CLI Strict Mode Flag.

Verifies that:
1. --strict enables strict mode in RuntimeConfig.
2. Strict mode causes unknown APIs to be marked with Escape Hatch comments.
3. Absence of --strict leaves code unmarked (passed through).
"""

from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.core.escape_hatch import EscapeHatch


def test_cli_strict_flag_enabled(tmp_path, capsys):
  """
  Scenario: User runs `convert --strict`.
  Expect: Unknown function is wrapped in failure marker.
  """
  infile = tmp_path / "strict_input.py"
  infile.write_text("y = torch.unknown_func(x)")
  outfile = tmp_path / "strict_output.py"

  # Run with --strict
  # Force source=torch to ensure rewriter checks 'torch.' prefix
  args = ["convert", str(infile), "--out", str(outfile), "--source", "torch", "--target", "jax", "--strict"]

  try:
    main(args)
  except SystemExit as e:
    assert e.code == 0

  assert outfile.exists()
  content = outfile.read_text()

  # Strict mode should bubble up error and attach comments
  # If this assertion fails, it means BaseRewriter isn't catching the unknown API.
  assert EscapeHatch.START_MARKER in content
  assert "API 'torch.unknown_func' not found" in content


def test_cli_strict_flag_disabled(tmp_path, capsys):
  """
  Scenario: User runs `convert` (without --strict).
  Expect: Unknown function passes through silenty.
  """
  infile = tmp_path / "lax_input.py"
  infile.write_text("y = torch.unknown_func(x)")
  outfile = tmp_path / "lax_output.py"

  # NO --strict flag
  args = ["convert", str(infile), "--out", str(outfile), "--source", "torch", "--target", "jax"]

  try:
    main(args)
  except SystemExit as e:
    assert e.code == 0

  assert outfile.exists()
  content = outfile.read_text()

  # Lax mode should NOT attach markers
  assert EscapeHatch.START_MARKER not in content
  # Code should persist
  assert "torch.unknown_func(x)" in content
