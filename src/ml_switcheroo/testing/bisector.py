"""Semantic Bisector: Automated Spec Repair.

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
import typing

from pydantic import JsonValue

from ml_switcheroo.exceptions import VerificationExecutionError
from ml_switcheroo.testing.runner import EquivalenceRunner

_UnpackedOpSpec: typing.TypeAlias = tuple[
  dict[str, JsonValue],
  list[str],
  dict[str, str],
  dict[str, dict[str, JsonValue]],
  str | None,
]


class SemanticsBisector:
  """Automated tool to find working constraints for an operation.

  Iterates through progressively relaxed constraints (e.g., tolerances) to find
  a configuration where the operation passes verification across frameworks.
  """

  def __init__(self, runner: EquivalenceRunner | None) -> None:
    """Initialize the bisector.

    Args:
        runner: The initialized EquivalenceRunner to execute verification.

    """
    self.runner = runner
    self.logger = logging.getLogger(__name__)

  def _verify_relaxation_step(
    self,
    runner: EquivalenceRunner,
    op_name: str,
    spec: _UnpackedOpSpec,
    rtol: float,
    atol: float,
  ) -> bool:
    """Execute a single tolerance relaxation verification step.

    Args:
        runner: Active ``EquivalenceRunner`` instance.
        op_name: Name of the operation being verified.
        spec: Unpacked ``(variants, params, hints, constraints, shape_calc)`` tuple.
        rtol: Candidate relative tolerance.
        atol: Candidate absolute tolerance.

    Returns:
        ``True`` if verification succeeded at ``(rtol, atol)``, otherwise ``False``.
    """
    variants, params, hints, constraints, output_shape_calc = spec
    try:
      success, _ = runner.verify(
        variants,
        params,
        hints=hints,
        constraints=constraints,
        shape_calc=output_shape_calc,
        rtol=rtol,
        atol=atol,
      )
      return bool(success)
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, VerificationExecutionError) as exc:
      self.logger.debug("Bisector relaxation step (%s, %s) failed for %s: %s", rtol, atol, op_name, exc)
      return False

  def propose_fix(self, op_name: str, op_def: typing.Mapping[str, JsonValue]) -> dict[str, JsonValue] | None:
    """Attempt to satisfy verification by mutating verification parameters.

    Strategies:
    1. Increase tolerances (1e-3 -> 1e-2 -> 1e-1).

    Args:
        op_name: Name of the operation.
        op_def: The definition dictionary from SemanticsManager.

    Returns:
        Dict: A patched version of op_def if a fix was found, else None.

    Raises:
        VerificationExecutionError: If ``self.runner`` is ``None``.
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
    raw_vars = op_def.get("variants", {})
    variants = raw_vars if isinstance(raw_vars, dict) else {}
    raw_std_args = op_def.get("std_args", [])
    std_args_raw = raw_std_args if isinstance(raw_std_args, list) else []
    raw_shape_calc = op_def.get("output_shape_calc")
    output_shape_calc = str(raw_shape_calc) if isinstance(raw_shape_calc, str) else None

    # Unpack info using runner helpers logic (simplified manual unpack here)
    params: list[str] = []
    hints: dict[str, str] = {}
    constraints: dict[str, dict[str, JsonValue]] = {}

    for arg in std_args_raw:
      if isinstance(arg, dict):
        name = str(arg.get("name", "unknown"))
        params.append(name)
        if "type" in arg:
          hints[name] = str(arg["type"])
        # Collect constraints
        cons: dict[str, JsonValue] = {}
        for k in ["min", "max", "options", "rank", "dtype", "shape_spec"]:
          if k in arg and arg[k] is not None:
            cons[k] = arg[k]
        if cons:
          constraints[name] = cons

      elif isinstance(arg, (list, tuple)) and len(arg) >= 1:
        name = str(arg[0])
        params.append(name)
        if len(arg) > 1:
          hints[name] = str(arg[1])

      elif isinstance(arg, str):
        params.append(arg)

    # Iterate steps
    if self.runner is None:
      raise VerificationExecutionError("SemanticsBisector requires an active EquivalenceRunner instance.")
    spec: _UnpackedOpSpec = (variants, params, hints, constraints, output_shape_calc)
    for rtol, atol in relaxation_steps:
      if self._verify_relaxation_step(self.runner, op_name, spec, rtol, atol):
        # If this passed, check if it's different from original defaults declared in op_def
        orig_rtol = op_def.get("test_rtol", 1e-3)
        orig_atol = op_def.get("test_atol", 1e-4)

        # Only propose a fix if values are different than current config
        if rtol != orig_rtol or atol != orig_atol:
          self.logger.info(f"Bisector found fix for {op_name}: rtol={rtol}, atol={atol}")
          patch: dict[str, JsonValue] = copy.deepcopy(dict(op_def))
          patch["test_rtol"] = rtol
          patch["test_atol"] = atol
          return patch

        # If it passed with settings matching current config (or standard default),
        # then no fix needed (false alarm or flaky test).
        return None

    return None
