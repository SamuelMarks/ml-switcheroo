"""Import ONNX Command Handler.

This module implements the `import-onnx` CLI command, which parses official
ONNX Markdown specifications and exports validated ODL YAML files for the
Knowledge Base.
"""

from pathlib import Path
from typing import Optional

from ml_switcheroo.importers.onnx_reader import OnnxSpecImporter
from ml_switcheroo.utils.console import log_error, log_info, log_success


def handle_import_onnx(
  spec_path: Path,
  out_dir: Optional[Path] = None,
  domain: str = "ai.onnx",
  opset_version: int = 21,
) -> int:
  """Handle the 'import-onnx' command.

  Parses an ONNX specification Markdown file and converts extracted operators
  into Operation Definition Language (ODL) YAML files.

  Args:
      spec_path: Path to the ONNX Markdown specification file (e.g. Operators.md).
      out_dir: Optional output directory to write discrete YAML files.
      domain: ONNX operator domain (e.g. 'ai.onnx', 'ai.onnx.ml').
      opset_version: ONNX opset version (e.g. 21).

  Returns:
      int: 0 on success, 1 on failure.
  """
  if not spec_path.exists():
    log_error(f"Specification file not found: {spec_path}")
    return 1

  importer = OnnxSpecImporter()
  parsed_ops = importer.parse_file(spec_path)
  if not parsed_ops:
    log_error(f"No valid operators extracted from {spec_path}")
    return 1

  odl_entries = importer.convert_to_odl(parsed_ops, domain=domain, opset_version=opset_version)

  if out_dir:
    count = importer.export_odl_yamls(odl_entries, out_dir)
    log_success(f"Successfully converted and exported {count} ONNX operators to {out_dir}")
  else:
    log_info(f"Successfully parsed {len(odl_entries)} ONNX operators from {spec_path.name}")

  return 0
