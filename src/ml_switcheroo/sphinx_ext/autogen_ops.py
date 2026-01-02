"""
Sphinx Hook for Automatic Operation Documentation Generation.

This module provides the logic to generate a dedicated documentation page (`.rst`)
for every Abstract Operation defined in the Semantic Knowledge Base.
It uses the `DocContextBuilder` and `OpPageRenderer` to produce rich,
interactive content including the Vertical Framework Tabs.

It is designed to be connected to the Sphinx `builder-inited` event.
"""

import os
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.util import logging

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.doc_context import DocContextBuilder
from ml_switcheroo.utils.doc_renderer import OpPageRenderer

logger = logging.getLogger(__name__)


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

  # Ensure directory exists
  out_dir.mkdir(parents=True, exist_ok=True)

  # Initialize Components
  semantics = SemanticsManager()
  builder = DocContextBuilder(semantics)
  renderer = OpPageRenderer()

  # Get all operations sorted alphabetically
  known_apis = semantics.get_known_apis()
  sorted_ops = sorted(known_apis.keys())

  if not sorted_ops:
    logger.warning("[ml-switcheroo] No operations found in semantics. Docs will be empty.")

  generated_files = []

  for op_name in sorted_ops:
    definition = known_apis[op_name]

    # Build Context
    context = builder.build(op_name, definition)

    # Render RST
    rst_content = renderer.render_rst(context)

    # Determine Filename (sanitize op_name just in case)
    safe_name = "".join(c for c in op_name if c.isalnum() or c in ("_", "-"))
    filename = f"{safe_name}.rst"
    file_path = out_dir / filename

    # Write File
    try:
      with open(file_path, "w", encoding="utf-8") as f:
        f.write(rst_content)
      generated_files.append(safe_name)
    except IOError as e:
      logger.warning(f"[ml-switcheroo] Failed to write {filename}: {e}")

  # Generate Index
  _write_index_file(out_dir, generated_files)
  logger.info(f"[ml-switcheroo] Generated {len(generated_files)} operation pages.")


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
    "Comprehensive reference of all abstract operations supported by the transpiler.",
    "",
    ".. toctree::",
    "   :maxdepth: 1",
    "   :caption: Operations",
    "",
  ]

  # Add file links
  for name in files:
    header.append(f"   {name}")

  content = "\n".join(header)

  try:
    with open(index_path, "w", encoding="utf-8") as f:
      f.write(content)
  except IOError as e:
    logger.warning(f"[ml-switcheroo] Failed to write index.rst: {e}")
