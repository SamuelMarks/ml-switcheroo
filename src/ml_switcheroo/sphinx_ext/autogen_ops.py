"""
Sphinx Hook for Automatic Operation Documentation Generation.

This module provides the logic to generate a dedicated documentation page (`.rst`)
for every Abstract Operation defined in the Semantic Knowledge Base.
It uses the `DocContextBuilder` and `OpPageRenderer` to produce rich,
interactive content including the Vertical Framework Tabs.

It is designed to be connected to the Sphinx `builder-inited` event.

Features:
1. Generates `.rst` pages for operations.
2. Exports `docs/operations.yaml` ensuring appended updates rather than destructive overwrites.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Union

from sphinx.application import Sphinx
from sphinx.util import logging

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.doc_context import DocContextBuilder
from ml_switcheroo.utils.doc_renderer import OpPageRenderer

logger = logging.getLogger(__name__)


class IndentedDumper(yaml.SafeDumper):
  """Custom Dumper to ensure lists are indented."""

  def increase_indent(self, flow=False, indentless=False):
    return super(IndentedDumper, self).increase_indent(flow, False)


def _build_yaml_entry(op_name: str, definition: Dict[str, Any]) -> Dict[str, Any]:
  """
  Normalizes internal semantics data into clean ODL YAML structure.
  """
  # 1. Normalize Arguments
  yaml_args = []
  raw_args = definition.get("std_args", [])

  for arg in raw_args:
    entry = {}
    if isinstance(arg, str):
      entry = {"name": arg, "type": "Any"}
    elif isinstance(arg, (list, tuple)) and len(arg) >= 2:
      entry = {"name": arg[0], "type": arg[1]}
    elif isinstance(arg, dict):
      entry = {k: v for k, v in arg.items() if v is not None}

    if entry:
      yaml_args.append(entry)

  # 2. Normalize Meta
  op_type = definition.get("op_type", "function")
  if hasattr(op_type, "value"):
    op_type = op_type.value

  # 3. Normalize Variants
  clean_variants = {}
  variants = definition.get("variants", {})
  for fw, details in variants.items():
    if details:
      clean_variants[fw] = dict(sorted(details.items()))

  return {
    "operation": op_name,
    "description": definition.get("description", "").strip(),
    "op_type": str(op_type),
    "std_args": yaml_args,
    "variants": clean_variants,
  }


def _write_yaml_update(out_path: Path, new_entries: List[Dict[str, Any]]) -> None:
  """
  Merges accumulated operations into the existing YAML file (Upsert logic).
  """
  existing_map = {}

  if out_path.exists():
    try:
      with open(out_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
        if isinstance(loaded, list):
          for item in loaded:
            if "operation" in item:
              existing_map[item["operation"]] = item
    except Exception as e:
      logger.warning(f"[ml-switcheroo] Could not read existing YAML: {e}. Overwriting.")

  # Upsert new entries (Code is source of truth during auto-gen)
  for entry in new_entries:
    existing_map[entry["operation"]] = entry

  final_list = list(existing_map.values())
  final_list.sort(key=lambda x: x.get("operation", ""))

  try:
    with open(out_path, "w", encoding="utf-8") as f:
      f.write(f"# {'=' * 78}\n")
      f.write(f"# ML-Switcheroo Operation Definitions (Auto-Generated)\n")
      f.write(f"# Filtered: Implemented by at least 2 frameworks\n")
      f.write(f"# {'=' * 78}\n\n")

      yaml.dump(
        final_list, f, Dumper=IndentedDumper, default_flow_style=False, sort_keys=False, allow_unicode=True, width=200
      )
    logger.info(f"[ml-switcheroo] Updated semantic YAML at {out_path}")
  except IOError as e:
    logger.warning(f"[ml-switcheroo] Failed to write YAML: {e}")


def generate_op_docs(app: Sphinx) -> None:
  """
  Sphinx Event Hook: Generates RST files for all operations.

  1.  Initializes the `SemanticsManager`.
  2.  Creates the `docs/ops` directory (cleaning it if safe).
  3.  Iterates through all known operations.
  4.  Builds the View Model using `DocContextBuilder`.
  5.  Renders RST using `OpPageRenderer`.
  6.  Writes `{op_name}.rst` files.
  7.  Generates `index.rst` to link all operations in a TOC.

  Args:
      app: The Sphinx application instance.
  """
  # Determine output directory
  # app.srcdir usually points to 'docs/'
  out_dir = Path(app.srcdir) / "ops"

  logger.info(f"[ml-switcheroo] Generating Operation Docs in {out_dir}")

  semantics = SemanticsManager()
  builder = DocContextBuilder(semantics)
  renderer = OpPageRenderer()

  known_apis = semantics.get_known_apis()
  sorted_ops = sorted(known_apis.keys())

  if not sorted_ops:
    logger.warning("[ml-switcheroo] No operations found.")

  generated_files = []
  yaml_entries = []

  for op_name in sorted_ops:
    definition = known_apis[op_name]
    variants = definition.get("variants", {})
    active = [v for v in variants.values() if v is not None]

    if len(active) < 2:
      continue

    yaml_entries.append(_build_yaml_entry(op_name, definition))

    context = builder.build(op_name, definition)
    rst_content = renderer.render_rst(context)
    safe_name = "".join(c for c in op_name if c.isalnum() or c in ("_", "-"))

    try:
      with open(ops_dir / f"{safe_name}.rst", "w", encoding="utf-8") as f:
        f.write(rst_content)
      generated_files.append(safe_name)
    except IOError:
      pass

  _write_index_file(ops_dir, generated_files)
  _write_yaml_update(docs_root / "operations.yaml", yaml_entries)


def _write_index_file(out_dir: Path, files: list[str]) -> None:
  """
  Generates the `index.rst` file linking all generated operations.

  Args:
      out_dir: The directory where the index file should be located.
      files: List of filenames (without extension) to include in the toctree.
  """
  index_path = out_dir / "index.rst"
  header = [
    "Operation Reference",
    "===================",
    "",
    ".. toctree::",
    "   :maxdepth: 1",
    "   :caption: Operations",
    "",
  ]
  for name in files:
    header.append(f"   {name}")

  with open(index_path, "w", encoding="utf-8") as f:
    f.write("\n".join(header))
