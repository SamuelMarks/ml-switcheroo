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
from typing import Dict, Any, Optional

from ml_switcheroo.testing.runner import EquivalenceRunner


class SemanticsBisector:
  """
  Automated tool to find working constraints for an operation.
  """

  def __init__(self, runner: EquivalenceRunner):
    """
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
    params = []
    hints = {}
    constraints = {}

    for arg in std_args_raw:
      if isinstance(arg, dict):
        name = arg.get("name", "unknown")
        params.append(name)
        if "type" in arg:
          hints[name] = arg["type"]
        # Collect constraints
        cons = {}
        for k in ["min", "max", "options", "rank", "dtype", "shape_spec"]:
          if k in arg and arg[k] is not None:
            cons[k] = arg[k]
        if cons:
          constraints[name] = cons

      elif isinstance(arg, (list, tuple)) and len(arg) >= 1:
        name = arg[0]
        params.append(name)
        if len(arg) > 1:
          hints[name] = arg[1]

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
          return None

      except Exception:
        continue

    return None
