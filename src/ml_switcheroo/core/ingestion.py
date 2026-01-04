"""
Ingestion logic for parsing source code into an Abstract Syntax Tree (AST).

This module handles the complexity of accepting different input formats (Python,
MLIR, TikZ, Custom DSLs) and normalizing them into a LibCST Module to be processed
by the core Transpilation Engine.
"""

from typing import Any, Optional

import libcst as cst

# MLIR Bridge
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.parser import MlirParser

# TikZ Bridge
from ml_switcheroo.core.tikz.parser import TikzParser
from ml_switcheroo.core.tikz.synthesizer import GraphSynthesizer
from ml_switcheroo.frameworks.base import FrameworkAdapter


def ingest_code(
  code: str,
  source_fw: str,
  target_fw: str,
  source_adapter: Optional[FrameworkAdapter],
  tracer: Any,
) -> cst.Module:
  """
  Parses input code handles non-python sources via adapters.

  Supports:
  1. Adapter-specific parsers (e.g. LaTeX).
  2. MLIR text parsing via `MlirParser`.
  3. TikZ text parsing via `TikzParser`.
  4. Standard Python parsing via `libcst`.

  Args:
      code: Raw source code string.
      source_fw: Key identifying the source framework (e.g., 'torch', 'tikz').
      target_fw: Key identifying the target framework (used for synthesis contexts).
      source_adapter: The loaded adapter for the source framework, if any.
      tracer: The trace logger instance for recording phase events.

  Returns:
      A validated Python LibCST Module.

  Raises:
      Exception: If parsing fails for any reason (syntax errors, etc.).
  """
  # 1. Adapter Hook (e.g. LatexParser)
  if source_adapter and hasattr(source_adapter, "create_parser"):
    tracer.start_phase("Custom Ingest", f"{source_fw} Parser")
    try:
      parser = source_adapter.create_parser(code)
      tree = parser.parse()
      tracer.log_mutation("Transformed Ingestion", "(Raw Source)", "(AST Parsed)")
      tracer.end_phase()
      return tree
    except Exception as e:
      tracer.end_phase()
      raise e

  # 2. MLIR source
  if source_fw == "mlir":
    tracer.start_phase("MLIR Ingest", "MLIR Text -> Python CST")
    try:
      parser = MlirParser(code)
      mlir_mod = parser.parse()
      gen = MlirToPythonGenerator()
      tree = gen.generate(mlir_mod)
      tracer.log_mutation("Ingestion", "(MLIR Text)", "(Python CST)")
      tracer.end_phase()
      return tree
    except Exception as e:
      tracer.end_phase()
      raise e

  # 3. TikZ source
  if source_fw == "tikz":
    tracer.start_phase("TikZ Ingest", "TikZ Text -> Logical Graph -> Python CST")
    try:
      parser = TikzParser(code)
      graph = parser.parse()
      # Determine synthesis flavour
      synth_target = "jax" if target_fw in ["jax", "flax", "flax_nnx"] else "torch"
      synthesizer = GraphSynthesizer(framework=synth_target)
      py_code = synthesizer.generate(graph, class_name="SwitcherooNet")
      # Parse the synthesized python code
      tree = cst.parse_module(py_code)
      tracer.log_mutation("Ingestion", "(TikZ Source)", f"(Python CST)\n{py_code}")
      tracer.end_phase()
      return tree
    except Exception as e:
      tracer.end_phase()
      raise e

  # 4. Standard Python
  tracer.start_phase("Preprocessing", "Parsing & Analysis")
  try:
    tree = cst.parse_module(code)
    tracer.log_mutation("Transformed Module", "(Raw Source)", "(AST Parsed)")
    tracer.end_phase()
    return tree
  except Exception as e:
    tracer.end_phase()
    raise e
