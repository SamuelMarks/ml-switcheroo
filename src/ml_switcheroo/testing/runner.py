"""Execution Engine for Semantics Verification (Hypothesis Integration).

Uses ``hypothesis`` to generate property-based test cases for operations.
Maps ODL definitions to Strategies and executes cross-framework comparison.
"""

import ast
import operator
import pkgutil
import typing

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ml_switcheroo.enums import CallableTargetProtocol, RuntimeTensorOrValue, ShapeProtocol
from ml_switcheroo.exceptions import VerificationExecutionError
from ml_switcheroo.frameworks import get_adapter
from ml_switcheroo.testing.fuzzer import InputFuzzer


def _eval_shape_builtin(name: str, args: list[RuntimeTensorOrValue]) -> RuntimeTensorOrValue:
  """Evaluate a whitelisted shape-calculation builtin call.

  Args:
      name: Builtin function identifier (``len``, ``tuple``, ``list``, ``int``, ``max``, ``min``).
      args: Evaluated positional arguments.

  Returns:
      Evaluated shape value.

  Raises:
      ValueError: If ``name`` is not a permitted shape builtin or arguments have invalid types.
  """
  if name == "len" and len(args) == 1 and isinstance(args[0], typing.Sized):
    return len(args[0])
  if name == "tuple" and len(args) == 1 and isinstance(args[0], typing.Iterable):
    return tuple(args[0])
  if name == "list" and len(args) == 1 and isinstance(args[0], typing.Iterable):
    return list(args[0])
  if name == "int" and len(args) == 1 and isinstance(args[0], (int, float)):
    return int(float(args[0]))
  if name in ("max", "min") and args:
    nums: list[int | float] = [a for a in args if isinstance(a, (int, float))]
    if len(nums) == len(args):
      return max(nums) if name == "max" else min(nums)
  raise ValueError(f"Unsupported call in shape_calc: {name}")


_SHAPE_BIN_OPS: dict[
  type[ast.operator],
  typing.Callable[[RuntimeTensorOrValue, RuntimeTensorOrValue], RuntimeTensorOrValue],
] = {
  ast.Add: operator.add,
  ast.Sub: operator.sub,
  ast.Mult: operator.mul,
  ast.FloorDiv: operator.floordiv,
  ast.Div: operator.truediv,
  ast.Mod: operator.mod,
}


def _is_shape_seq(val: RuntimeTensorOrValue) -> typing.TypeGuard[typing.Sequence[int | float]]:
  """Check if a value is an indexable sequence of shape dimensions.

  Args:
      val: Value evaluated during shape calculation.

  Returns:
      True if ``val`` is a sequence of numeric dimensions.
  """
  return isinstance(val, typing.Sequence)


def _eval_shape_attribute(node: ast.Attribute, param_name: str, x_val: RuntimeTensorOrValue) -> RuntimeTensorOrValue:
  """Evaluate safe ``.shape`` attribute access within a ``shape_calc`` expression.

  Args:
      node: AST ``Attribute`` node.
      param_name: Lambda parameter identifier bound to ``x_val``.
      x_val: Input tensor or array value.

  Returns:
      Shape sequence extracted from the target value.

  Raises:
      VerificationExecutionError: If the attribute is private or not ``shape`` on a ``ShapeProtocol`` value.
  """
  val = _eval_node(node.value, param_name, x_val)
  if node.attr.startswith("_"):
    raise VerificationExecutionError(f"Private attribute access forbidden: {node.attr}")
  if node.attr != "shape" or not isinstance(val, ShapeProtocol):
    raise VerificationExecutionError(f"Unsupported attribute '{node.attr}' in shape_calc.")
  return val.shape


def _eval_shape_subscript(node: ast.Subscript, param_name: str, x_val: RuntimeTensorOrValue) -> RuntimeTensorOrValue:
  """Evaluate sequence indexing or slicing within a ``shape_calc`` expression.

  Args:
      node: AST ``Subscript`` node.
      param_name: Lambda parameter identifier bound to ``x_val``.
      x_val: Input tensor or array value.

  Returns:
      Indexed dimension or sliced sub-sequence.

  Raises:
      VerificationExecutionError: If the target is not a sequence or indices/bounds are non-integers.
  """
  val = _eval_node(node.value, param_name, x_val)
  if not _is_shape_seq(val):
    raise VerificationExecutionError("Subscript target in shape_calc must be a shape sequence.")
  if isinstance(node.slice, ast.Slice):
    lower = _eval_node(node.slice.lower, param_name, x_val) if node.slice.lower else None
    upper = _eval_node(node.slice.upper, param_name, x_val) if node.slice.upper else None
    step = _eval_node(node.slice.step, param_name, x_val) if node.slice.step else None
    if not (
      (lower is None or isinstance(lower, int))
      and (upper is None or isinstance(upper, int))
      and (step is None or isinstance(step, int))
    ):
      raise VerificationExecutionError("Slice bounds in shape_calc must evaluate to integers or None.")
    return val[slice(lower, upper, step)]
  slc = _eval_node(node.slice, param_name, x_val)
  if not isinstance(slc, int):
    raise VerificationExecutionError("Subscript index in shape_calc must evaluate to an integer.")
  return val[slc]


def _eval_node(node: ast.AST, param_name: str, x_val: RuntimeTensorOrValue) -> RuntimeTensorOrValue:
  """Recursively evaluate a safe subset of Python AST nodes for ``shape_calc``.

  Args:
      node: AST node to evaluate.
      param_name: Lambda parameter identifier bound to ``x_val``.
      x_val: Input tensor or array value.

  Returns:
      Evaluated Python value.

  Raises:
      VerificationExecutionError: If unary negation receives a non-numeric operand.
      ValueError: If an unknown identifier or unsupported AST expression is encountered.
  """
  if isinstance(node, ast.Constant):
    val_const = node.value
    return str(val_const) if isinstance(val_const, bytes) else val_const
  if isinstance(node, ast.Name):
    if node.id == param_name:
      return x_val
    raise ValueError(f"Unknown identifier in shape_calc: {node.id}")
  if isinstance(node, ast.Tuple):
    return tuple(_eval_node(elt, param_name, x_val) for elt in node.elts)
  if isinstance(node, ast.List):
    return [_eval_node(elt, param_name, x_val) for elt in node.elts]
  if isinstance(node, ast.Attribute):
    return _eval_shape_attribute(node, param_name, x_val)
  if isinstance(node, ast.Subscript):
    return _eval_shape_subscript(node, param_name, x_val)
  if isinstance(node, ast.BinOp):
    op_fn = _SHAPE_BIN_OPS.get(type(node.op))
    if op_fn is not None:
      return op_fn(_eval_node(node.left, param_name, x_val), _eval_node(node.right, param_name, x_val))
  if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
    op_val = _eval_node(node.operand, param_name, x_val)
    if not isinstance(op_val, (int, float, complex)):
      raise VerificationExecutionError(f"Unary '-' requires numeric operand in shape_calc, got {type(op_val).__name__}.")
    neg_val: RuntimeTensorOrValue = operator.neg(op_val)
    return neg_val
  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
    return _eval_shape_builtin(node.func.id, [_eval_node(a, param_name, x_val) for a in node.args])
  raise ValueError(f"Unsupported AST node in shape_calc: {type(node).__name__}")


def _evaluate_shape_calc(shape_calc: str, x_val: RuntimeTensorOrValue) -> RuntimeTensorOrValue:
  """Safely evaluate a restricted shape calculation lambda expression AST on ``x_val``.

  Supports ``lambda <arg>: <expr>`` where ``<expr>`` consists of constants, tuples, lists,
  attribute access (e.g., ``x.shape``), subscripts, and basic arithmetic operations.

  Args:
      shape_calc: String representation of the shape lambda (e.g. ``'lambda x: x.shape'``).
      x_val: Input array/tensor passed to the lambda parameter.

  Returns:
      The computed expected shape tuple or value.

  Raises:
      ValueError: If the expression is not a valid restricted lambda AST.
  """
  tree = ast.parse(shape_calc.strip(), mode="eval")
  if not isinstance(tree.body, ast.Lambda) or len(tree.body.args.args) != 1:
    raise ValueError(f"Expected single-argument lambda in shape_calc, got: {shape_calc}")
  param_name = tree.body.args.args[0].arg
  return _eval_node(tree.body.body, param_name, x_val)


class EquivalenceRunner:
  """Orchestrate cross-framework equivalence testing for machine learning operations.

  This class manages property-based testing and input generation via Hypothesis.
  It generates randomized inputs matching parameter specifications, routes those
  inputs to multiple framework implementations, and verifies that the outputs are
  numerically equivalent and maintain expected structural traits (e.g., shapes).
  """

  def __init__(self) -> None:
    """Initialize the equivalence runner with an input fuzzer.

    Sets up the underlying InputFuzzer to generate randomized input strategies
    according to variable/parameter constraints.
    """
    self.fuzzer = InputFuzzer()

  def verify(
    self,
    variants: typing.Mapping[str, RuntimeTensorOrValue],
    params: list[str],
    hints: typing.Mapping[str, str] | None = None,
    constraints: typing.Mapping[str, typing.Mapping[str, RuntimeTensorOrValue]] | None = None,
    shape_calc: str | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-4,
  ) -> tuple[bool, str]:
    """Run property-based verification across multiple framework variants using Hypothesis.

    Generates test cases dynamically and evaluates whether all registered framework
    variants produce equivalent numerical outputs given identical inputs.

    Args:
        variants: A dictionary mapping framework names to their details (API paths and argument mappings).
        params: A list of parameter names to generate strategies for.
        hints: Optional dictionaries providing typing or generation hints for the parameters.
        constraints: Optional constraint structures to bound generated values.
        shape_calc: Optional string representation of a shape calculation lambda/function for checking outputs.
        rtol: Relative tolerance for floating point comparisons.
        atol: Absolute tolerance for floating point comparisons.

    Returns:
        A tuple of (success_boolean, status_or_error_message).
    """
    # Build composite strategy
    strat_dict = self.fuzzer.build_strategies(params, hints, constraints)

    # State to track failure messages from inside the hypothesis loop
    failure_msg = []

    @settings(max_examples=20, deadline=None)
    @given(st.fixed_dictionaries(strat_dict))
    def run_check(inputs: dict[str, RuntimeTensorOrValue]) -> None:
      """Execute a single property-based test iteration using generated inputs.

      Runs the operation on all defined framework variants and performs equivalence
      assertions on their results.

      Args:
          inputs: The generated input dictionary mapping parameter names to values.

      Returns:
          None.
      """
      # Shape Check (Feature 20)
      if shape_calc and len(inputs) > 0 and len(params) > 0:
        # Basic shape check simulation on input
        # This is tricky inside hypothesis loop without execution results yet,
        # usually shape check is post-execution.
        pass

      results: dict[str, RuntimeTensorOrValue] = {}
      # Execution Loop
      for fw, details in variants.items():
        if not isinstance(details, dict) or "api" not in details:
          continue

        try:
          # Pivot Arguments
          raw_args = details.get("args", {})
          args_map: dict[str, str] = (
            {str(k): str(v) for k, v in raw_args.items() if v is not None} if isinstance(raw_args, dict) else {}
          )
          fw_args = self._remap_args(inputs, args_map)
          # Adapt Inputs
          fw_ready = self.fuzzer.adapt_to_framework(fw_args, fw)
          # Run
          res = self._execute_api(str(details["api"]), fw_ready)
          # Normalize Output
          adp = get_adapter("numpy")
          results[fw] = adp.convert(res) if adp else res
        except (ImportError, ModuleNotFoundError):
          # Optional target framework is not installed in the current environment
          pass
        except (TypeError, ValueError, AttributeError, RuntimeError, KeyError, IndexError, ZeroDivisionError) as e:
          failure_msg.append(f"Crash in {fw}: {e}")

      # Post-Execution Shape Check
      if shape_calc and "x" in inputs:
        try:
          expected_shape = _evaluate_shape_calc(shape_calc, inputs["x"])

          # Verify results
          for r in results.values():
            r_shape = r.shape if isinstance(r, ShapeProtocol) else None
            if r_shape is not None:
              s = tuple(r_shape) if isinstance(r_shape, (list, tuple)) else (r_shape,)
              exp = tuple(expected_shape) if isinstance(expected_shape, (list, tuple)) else (expected_shape,)
              if s != exp:
                failure_msg.append(f"Shape Mismatch: {s} != {exp}")
        except (SyntaxError, ValueError, TypeError, AttributeError, ZeroDivisionError, IndexError, KeyError) as e:
          failure_msg.append(f"Shape Calculation Error: {e}")

      # Comparison
      self._compare_results(results, rtol, atol, failure_msg)

    try:
      run_check()
      if failure_msg:
        # Return the LAST failure (often most relevant)
        return False, f"Failures Detected: {failure_msg[-1]}"
      return True, "✅ Verified"
    except (AssertionError, VerificationExecutionError, TypeError, ValueError, RuntimeError) as e:
      # Hypothesis raises explicit errors when assertions fail
      return False, f"Verification Failed: {e}"

  def _execute_api(self, api: str, kwargs: typing.Mapping[str, RuntimeTensorOrValue]) -> RuntimeTensorOrValue:
    """Dynamically imports and calls a framework API function with specified arguments.

    Args:
        api: Fully qualified module and function name string (e.g. 'NumPy.add').
        kwargs: Dictionary of keyword arguments to pass to the imported function.

    Returns:
        The return value of the executed API function, or None if the API path is invalid.

    Raises:
        TypeError: If the resolved attribute is not callable.
    """
    if "." not in api:
      return None
    m, f = api.rsplit(".", 1)
    resolved = pkgutil.resolve_name(f"{m}:{f}")
    if not isinstance(resolved, CallableTargetProtocol):
      raise TypeError(f"Resolved API target '{api}' is not callable.")
    res: RuntimeTensorOrValue = resolved(**kwargs)
    return res

  def _remap_args(
    self, inputs: typing.Mapping[str, RuntimeTensorOrValue], mapping: typing.Mapping[str, str]
  ) -> dict[str, RuntimeTensorOrValue]:
    """Remap input argument names to match the expected parameter names of a framework variant.

    Args:
        inputs: Dictionary of generated input parameters and their values.
        mapping: Dictionary mapping generated input names to framework-specific names.

    Returns:
        A new dictionary with parameter names remapped to framework-specific names.
    """
    return {mapping.get(k, k): v for k, v in inputs.items()}

  def _compare_results(
    self,
    results: typing.Mapping[str, RuntimeTensorOrValue],
    rtol: float,
    atol: float,
    err_box: list[str],
  ) -> None:
    """Compare execution results from different frameworks and records mismatches.

    Performs exhaustive pairwise deep comparisons between the outputs of all
    frameworks, raising an AssertionError and storing messages if any mismatches
    are encountered.

    Args:
        results: Dictionary mapping framework names to their execution results.
        rtol: Relative tolerance for numerical values.
        atol: Absolute tolerance for numerical values.
        err_box: A list used to accumulate error messages for reporting.

    Raises:
        AssertionError: If any of the results differ significantly.
    """
    if len(results) < 2:
      return
    vals = list(results.values())
    ref = vals[0]
    fw_keys = list(results.keys())
    ref_fw = fw_keys[0]

    for i, v in enumerate(vals[1:], 1):
      current_fw = fw_keys[i]
      if not self._deep_compare(ref, v, rtol, atol):
        m = f"Mismatch: {ref_fw}({ref}) vs {current_fw}({v})"
        err_box.append(m)
        raise AssertionError(m)

  def _deep_compare(
    self,
    a: RuntimeTensorOrValue,
    b: RuntimeTensorOrValue,
    rtol: float = 1e-3,
    atol: float = 1e-4,
  ) -> bool:
    """Recursively checks two values for structural and numerical equivalence.

    Handles lists, tuples, scalar numbers, NumPy arrays, and other types with custom
    tolerance-based comparison for float-like structures.

    Args:
        a: The first value/object to compare.
        b: The second value/object to compare.
        rtol: Relative tolerance for numeric equivalence.
        atol: Absolute tolerance for numeric equivalence.

    Returns:
        True if the values are structurally and numerically equivalent, False otherwise.
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
      return len(a) == len(b) and all(self._deep_compare(x, y, rtol, atol) for x, y in zip(a, b, strict=False))

    if isinstance(a, (int, float, np.ndarray, np.generic)):
      try:
        a_arr = np.asanyarray(a)
        b_arr = np.asanyarray(b)
        if a_arr.shape != b_arr.shape:
          return False
        # Handle string/object types safely
        if a_arr.dtype.kind in ["U", "S", "O"]:
          return np.array_equal(a_arr, b_arr)
        return np.allclose(a_arr, b_arr, rtol=rtol, atol=atol, equal_nan=True)
      except (TypeError, ValueError):
        return False
    return bool(a == b)
