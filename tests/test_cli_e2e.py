"""Auto-generated doc."""

from ml_switcheroo.cli.__main__ import main
from unittest.mock import patch, MagicMock


def test_cli_e2e_suggest(tmp_path):
  """Auto-generated doc."""
  try:
    main(["suggest", "torch.nn.Linear", "--out", str(tmp_path)])
  except SystemExit:
    pass


@patch("ml_switcheroo.cli.handlers.convert.ASTEngine")
def test_cli_e2e_convert(mock_engine, tmp_path):
  """Auto-generated doc."""
  out = tmp_path / "out_convert"
  out.mkdir()

  in_file = tmp_path / "model.py"
  in_file.write_text("import torch; torch.nn.Linear(10, 10)\n")

  # ASTEngine is instantiated and convert() is called
  mock_instance = MagicMock()
  mock_engine.return_value = mock_instance
  mock_res = MagicMock()
  mock_res.code = "import jax"
  mock_instance.convert.return_value = mock_res

  try:
    main(["convert", str(in_file), "--out", str(out), "--source", "torch", "--target", "jax"])
  except SystemExit:
    pass


@patch("ml_switcheroo.cli.handlers.verify.BatchValidator.run_all", return_value={})
def test_cli_e2e_ci(mock_run, tmp_path):
  """Auto-generated doc."""
  out = tmp_path / "report.json"
  try:
    main(["ci", "--json-report", str(out)])
  except SystemExit:
    pass


def test_cli_e2e_docs(tmp_path):
  """Auto-generated doc."""
  out = tmp_path / "MIGRATION.md"
  try:
    main(["gen-docs", "--out", str(out)])
  except SystemExit:
    pass


def test_cli_e2e_matrix(tmp_path):
  """Auto-generated doc."""
  try:
    main(["matrix"])
  except SystemExit:
    pass


def test_cli_e2e_audit(tmp_path):
  """Auto-generated doc."""
  in_file = tmp_path / "model.py"
  in_file.write_text("import torch\nclass Model: pass\n")
  try:
    main(["audit", str(in_file)])
  except SystemExit:
    pass


@patch("ml_switcheroo.generated_tests.generator.get_template", return_value=False)
def test_cli_e2e_gen_tests(mock_get_template, tmp_path):
  """Auto-generated doc."""
  try:
    main(["gen-tests"])
  except SystemExit:
    pass


def test_cli_e2e_weight_script(tmp_path):
  """Auto-generated doc."""
  in_file = tmp_path / "model.py"
  in_file.write_text("import torch\nclass Model: pass\n")
  out = tmp_path / "weight.py"
  try:
    main(["gen-weight-script", str(in_file), "--out", str(out)])
  except SystemExit:
    pass
