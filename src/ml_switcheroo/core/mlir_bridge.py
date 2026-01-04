"""
MLIR Round-trip Verification Logic.

This module provides utilities to verify structural integrity by converting
Python ASTs to MLIR and back. These functions are separated to keep the main
engine logic focused on high-level orchestration.
"""

from typing import Any

import libcst as cst

from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.parser import MlirParser


def run_mlir_roundtrip(tree: cst.Module, tracer: Any) -> cst.Module:
  """
  Executes CST -> MLIR Text -> CST pipeline for verification.

  Used when intermediate="mlir" is selected to verify the structural integrity
  of the MLIR bridge. Captures and logs exceptions to the tracer without
  crashing the pipeline.

  Args:
      tree: The Python CST to verify.
      tracer: The active trace logger for recording the phases and any mutations
          or warnings.

  Returns:
      A reconstructed Python CST if successful, or the original tree if the
      roundtrip fails (ensuring fail-safe behavior).
  """
  try:
    tracer.start_phase("MLIR Bridge", "CST -> MLIR Text -> CST")
    emitter = PythonToMlirEmitter()
    mlir_cst = emitter.convert(tree)
    mlir_text = mlir_cst.to_text()
    tracer.log_mutation("MLIR Generation", "(Python CST)", mlir_text)

    parser = MlirParser(mlir_text)
    mlir_cst_restored = parser.parse()

    generator = MlirToPythonGenerator()
    restored_tree = generator.generate(mlir_cst_restored)
    tracer.end_phase()
    return restored_tree
  except Exception as e:
    tracer.log_warning(f"MLIR Bridge Failed: {e}")
    tracer.end_phase()
    return tree
