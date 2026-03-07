"""
Semantics Bisector: Automated Spec Repair.

This module implements a feedback loop ("Bisector") that attempts to fix failing
verification tests by incrementally relaxing the constraints in the ODL specifications.
It closes the loop between Testing and Definition.

Capabilities:

1.  **Tolerance Relaxation**: If a test fails on numeric comparison, the bisector
    retries with looser `rtol`/`atol` values.
2.  **Suggestion Generation**: Returns a mutated OperationDef dict that passes tests,
    which can be used to patch the Knowledge Base.
"""

import copy
import logging
from typing import Dict, Any, Optional, List

from ml_switcheroo.testing.runner import EquivalenceRunner


class SemanticsBisector:
  """
  Automated tool to find working constraints for an operation.

  Iterates through progressively relaxed constraints (e.g., tolerances) to find
  a configuration where the operation passes verification across frameworks.
  """

  def __init__(self, runner: EquivalenceRunner) -> None:
    """
    Initialize the bisector.

    Args:
        runner: The initialized EquivalenceRunner to execute verification.
    """
    self.runner = runner
    self.logger = logging.getLogger(__name__)

  def propose_fix(self, op_name: str, op_def: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Attempts to satisfy verification by mutating verification parameters.

    Strategies:
    1. Increase tolerances (1e-3 -> 1e-2 -> 1e-1).

    Args:
        op_name: Name of the operation.
        op_def: The definition dictionary from SemanticsManager.

    Returns:
        Dict: A patched version of op_def if a fix was found, else None.
    """
    # Standard Relaxation Ladder
    # (rtol, atol) tuples
    relaxation_steps = [
      (1e-3, 1e-4),  # Standard
      (1e-3, 1e-3),  # Loose Absolute
      (1e-2, 1e-3),  # Loose Relative
      (0.05, 0.05),  # Approximate (e.g. for fast-math approxs)
      (0.1, 0.1),  # Very Loose
    ]

    # Extract static info
    variants = op_def.get("variants", {})
    std_args_raw = op_def.get("std_args", [])
    output_shape_calc = op_def.get("output_shape_calc")

    # Unpack info using runner helpers logic (simplified manual unpack here)
    params: List[str] = []
    hints: Dict[str, str] = {}
    constraints: Dict[str, Dict[str, Any]] = {}

    for arg in std_args_raw:
      if isinstance(arg, dict):
        name = arg.get("name", "unknown")  # pragma: no cover
        params.append(name)  # pragma: no cover
        if "type" in arg:  # pragma: no cover
          hints[name] = str(arg["type"])  # pragma: no cover
        # Collect constraints
        cons = {}  # pragma: no cover
        for k in ["min", "max", "options", "rank", "dtype", "shape_spec"]:  # pragma: no cover
          if k in arg and arg[k] is not None:  # pragma: no cover
            cons[k] = arg[k]  # pragma: no cover
        if cons:  # pragma: no cover
          constraints[name] = cons  # pragma: no cover

      elif isinstance(arg, (list, tuple)) and len(arg) >= 1:
        name = str(arg[0])  # pragma: no cover
        params.append(name)  # pragma: no cover
        if len(arg) > 1:  # pragma: no cover
          hints[name] = str(arg[1])  # pragma: no cover

      elif isinstance(arg, str):
        params.append(arg)

    # Iterate steps
    for rtol, atol in relaxation_steps:
      try:
        success, _ = self.runner.verify(
          variants,
          params,
          hints=hints,
          constraints=constraints,
          shape_calc=output_shape_calc,
          rtol=rtol,
          atol=atol,
        )

        if success:
          # If this passed, check if it's different from original defaults declared in op_def
          orig_rtol = op_def.get("test_rtol", 1e-3)
          orig_atol = op_def.get("test_atol", 1e-4)

          # Only propose a fix if values are different than current config
          if rtol != orig_rtol or atol != orig_atol:
            self.logger.info(f"Bisector found fix for {op_name}: rtol={rtol}, atol={atol}")
            patch = copy.deepcopy(op_def)
            patch["test_rtol"] = rtol
            patch["test_atol"] = atol
            return patch

          # If it passed with settings matching current config (or standard default),
          # then no fix needed (false alarm or flaky test).
          return None  # pragma: no cover

      except Exception:  # pragma: no cover
        continue  # pragma: no cover

    return None
