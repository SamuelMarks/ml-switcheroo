"""
Batch Validation Runner.

Iterates over all defined semantic operations and verifys their correctness.
Prioritizes "Manual/Human" tests found on disk over "Auto/Robotic" fuzzing.
This closes the loop for Workflow B, allowing manual fixes to count as success.

Updated to handle Rich Parameter Constraints (dict items in std_args),
including Dtype and Rank constraints, and **Input-Dependent Output Shape** verification.
"""

import ast
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Tuple

from rich.progress import track

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.runner import EquivalenceRunner


class BatchValidator:
  """
  Orchestrates the validation of the entire knowledge base.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initializes the validator.

    Args:
        semantics: The loaded SemanticsManager containing API definitions.
    """
    self.semantics = semantics
    self.runner = EquivalenceRunner()

  def run_all(self, verbose: bool = False, manual_test_dir: Optional[Path] = None) -> Dict[str, bool]:
    """
    Runs verification for all known APIs.

    Validation Hierarchy:

    1. **Manual Test Priority**: If a file contains `def test_<op_name>():`,
       we assume the developer has manually verified this operation. It is
       marked as Passing.
    2. **Automated Fuzzing**: If no manual test exists, we unpack the
       arguments, type hints, and constraints from the spec, generate random inputs,
       and check equivalence using `EquivalenceRunner`.

    Args:
        verbose: Show progress bar using Rich.
        manual_test_dir: Directory containing manual tests to scan (default: None).

    Returns:
        Dict[op_name, bool]: Validation status (True=Pass, False=Fail).
    """
    results = {}
    known_apis = self.semantics.get_known_apis()
    op_names = sorted(known_apis.keys())

    # Pre-scan for manual tests to avoid repeated IO during iteration
    manual_tests = set()
    if manual_test_dir:
      manual_tests = self._scan_manual_tests(manual_test_dir)

    iterator = op_names
    if verbose:
      iterator = track(op_names, description="🧪 Verifying Semantics...")

    for op_name in iterator:
      # 1. Manual Test Priority
      if op_name in manual_tests:
        # Presence of a manual test implies Human Verification.
        results[op_name] = True
        continue

      # 2. Automated Fuzzing
      details = known_apis[op_name]
      variants = details.get("variants", {})
      std_args_raw = details.get("std_args", ["x"])
      output_shape_calc = details.get("output_shape_calc")

      # Unpack typed arguments and constraints
      params, hints, constraints = self._unpack_args(std_args_raw)

      passed, _ = self.runner.verify(variants, params, hints=hints, constraints=constraints, shape_calc=output_shape_calc)
      results[op_name] = passed

    return results

  def _unpack_args(self, raw_args: List[Any]) -> Tuple[List[str], Dict[str, str], Dict[str, Dict]]:
    """
    Separates argument names from type hints and extracts semantic constraints.

    Handles formats:
    1. Legacy strings: `["x", "axis"]`
    2. Tuple types: `[("x", "Array"), ("axis", "int")]`
    3. Rich Dictionaries: `[{"name": "x", "min": 0}, {"name": "axis", "type": "int"}]`

    Returns:
        A tuple containing:
            - List of argument names [str].
            - Dictionary of type hints {name: type_str}.
            - Dictionary of constraints {name: {min: val, dtype: 'int64', ...}}.
    """
    params = []
    hints = {}
    constraints = {}

    for item in raw_args:
      # Case 3: Rich Dict (ODL)
      if isinstance(item, dict):
        name = item.get("name")
        if not name:
          continue  # Malformed
        params.append(name)

        if "type" in item:
          hints[name] = item["type"]

        # Extract constraints if present (Updated for dtype/rank)
        constrs = {}
        for k in ["min", "max", "options", "rank", "dtype"]:
          if k in item and item[k] is not None:
            constrs[k] = item[k]

        if constrs:
          constraints[name] = constrs

      # Case 2: Typed Tuple
      elif isinstance(item, (list, tuple)) and len(item) == 2:
        name, annotation = item
        params.append(name)
        hints[name] = annotation

      # Case 1: Simple String
      elif isinstance(item, str):
        params.append(item)
        # No hint available

    return params, hints, constraints

  def _scan_manual_tests(self, root: Path) -> Set[str]:
    """
    Scans python files in root for test functions matching op names.

    Looks for: `def test_<op_name>():` in non-generated files.

    Args:
        root: Directory to search recursively.

    Returns:
        Set of operation names found in manual tests.
    """
    found = set()
    if not root.exists():
      return found

    for py_file in root.rglob("*.py"):
      # Skip generated files (robotic tests) to ensure we only count Human work
      if "generated" in py_file.parts:
        continue

      try:
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for node in tree.body:
          if (
            isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and not node.name.startswith("test_gen_")
          ):
            # Extract 'matmul' from 'test_matmul'
            op = node.name[5:]
            found.add(op)
      except Exception:
        # Ignore parse errors in user files
        pass
    return found
