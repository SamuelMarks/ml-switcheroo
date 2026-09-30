"""Test suite for import-onnx CLI handler."""

from pathlib import Path

from ml_switcheroo.cli.handlers.import_onnx import handle_import_onnx

SAMPLE_MD = (
  '### <a name="Relu"></a>**Relu**\n\n'
  "#### Summary\n\n"
  "Applies relu activation.\n\n"
  "#### Inputs\n\n"
  "<dl><dt>X : T</dt><dd>input</dd></dl>\n"
)


def test_handle_import_onnx_missing_file(tmp_path: Path) -> None:
  """Verifies error exit when spec file does not exist."""
  assert handle_import_onnx(tmp_path / "missing.md") == 1


def test_handle_import_onnx_empty_file(tmp_path: Path) -> None:
  """Verifies error exit when spec file contains no valid operators."""
  p = tmp_path / "empty.md"
  p.write_text("# Title\n\nNo operators here.")
  assert handle_import_onnx(p) == 1


def test_handle_import_onnx_success_no_out_dir(tmp_path: Path) -> None:
  """Verifies successful parsing without writing files when out_dir is None."""
  p = tmp_path / "Operators.md"
  p.write_text(SAMPLE_MD)
  assert handle_import_onnx(p, out_dir=None) == 0


def test_handle_import_onnx_success_with_out_dir(tmp_path: Path) -> None:
  """Verifies successful parsing and exporting of discrete ODL YAML files."""
  p = tmp_path / "Operators.md"
  p.write_text(SAMPLE_MD)
  out_dir = tmp_path / "exported_odl"
  assert handle_import_onnx(p, out_dir=out_dir, domain="ai.onnx", opset_version=21) == 0
  assert out_dir.exists()
  assert (out_dir / "Relu.yaml").exists()
  yaml_content = (out_dir / "Relu.yaml").read_text()
  assert "Relu" in yaml_content
  assert "ai.onnx.Relu" in yaml_content
