"""
Tests for Argument Normalization Logic in PivotRewriter.

Verifies that the Rewriter correctly pivots argument names and positions
based on the abstract specification.

Coverage:
    1. Keyword argument renaming (e.g., `dim` -> `axis`, `input` -> `a`).
    2. Positional argument preservation and alignment.
    3. Mixed usage (positional + keyword).
    4. Passthrough of unknown/extra arguments (kwargs).
    5. Handling of Typed specifications (`("x", "int")`).
    6. Argument Injection (Feature: inject_args).
    7. Argument Value Mapping (Feature: arg_values).
    8. **Dynamic Module Detection** (Replacing hardcoded sets).
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockArgSemantics(SemanticsManager):
  """
  Mock Data for Argument Normalization tests.
  Provides specific scenarios for identifying renaming, reordering, and passthrough.
  """

  def __init__(self) -> None:
    """Initializes with specific argument mapping scenarios."""
    self.data = {}
    self.import_data = {}
    self._reverse_index = {}
    self._key_origins = {}

    # NEW: Populate framework configs to support dynamic module detection
    self.framework_configs = {
      "torch": {"alias": {"module": "torch", "name": "torch"}},
      "jax": {"alias": {"module": "jax.numpy", "name": "jnp"}},
      "experimental_fw": {"alias": {"module": "exp.net", "name": "exp"}},
    }

    # 1. Complex Renaming ('sum')
    # Source: sum(input, dim)
    # Standard: sum(x, axis)
    # Target: sum(a, axis)
    self._inject_op(
      op_name="sum",
      std_args=["x", "axis"],
      variants={
        "torch": {
          "api": "torch.sum",
          "args": {"x": "input", "axis": "dim"},
        },
        "jax": {
          "api": "jax.numpy.sum",
          "args": {"x": "a", "axis": "axis"},
        },
      },
    )

    # 2. Simple Positional ('div')
    self._inject_op(
      op_name="div",
      std_args=["x", "y"],
      variants={
        "torch": {"api": "torch.div"},
        "jax": {"api": "jax.numpy.divide"},
      },
    )

    # 3. Typed Arguments Specification ('randint')
    # Standard: randint(low: int, high: int)
    self._inject_op(
      op_name="randint",
      std_args=[("low", "int"), ("high", "int")],
      variants={
        "torch": {"api": "torch.randint"},
        "jax": {"api": "jax.random.randint"},
      },
    )

    # 4. Injection Scenario ('normalize')
    # Target needs epsilon injected
    self._inject_op(
      op_name="normalize",
      std_args=["x"],
      variants={
        "torch": {"api": "torch.normalize"},
        "jax": {"api": "jax.nn.normalize", "inject_args": {"epsilon": 1e-5, "flag": True}},
      },
    )

    # 5. Value Mapping Scenario ('reduce')
    # torch.reduce(..., val='mean') -> jax.reduce(..., mode='avg')
    # torch.reduce(..., val=0)      -> jax.reduce(..., mode='none')
    self._inject_op(
      op_name="reduce",
      std_args=["x", "val"],
      variants={
        "torch": {"api": "torch.reduce", "args": {"val": "reduction"}},
        "jax": {"api": "jax.reduce", "args": {"val": "mode"}, "arg_values": {"val": {"mean": "'avg'", "0": "'none'"}}},
      },
    )

    # 6. Method Style Test ('method_op')
    # Used to verify if _is_module_alias prevents receiver injection
    self._inject_op(
      op_name="method_op",
      std_args=["x", "y"],
      variants={
        "torch": {"api": "torch.method_op"},
        "jax": {"api": "jax.method_op"},
      },
    )

  def _inject_op(self, op_name, std_args, variants):
    self.data[op_name] = {"std_args": std_args, "variants": variants}
    for _, details in variants.items():
      if "api" in details:
        self._reverse_index[details["api"]] = (op_name, self.data[op_name])


@pytest.fixture
def engine() -> PivotRewriter:
  """Returns a Rewriter for Torch -> JAX arg testing."""
  semantics = MockArgSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Parses code, applies rewriter, and returns generated code."""
  tree = cst.parse_module(code)
  return tree.visit(rewriter).code


def test_keyword_translation(engine: PivotRewriter) -> None:
  """
  Test that keyword arguments are renamed correctly.

  Input:  `torch.sum(input=z, dim=1)`
  Logic:  input -> x -> a, dim -> axis -> axis.
  Output: `jax.numpy.sum(a=z, axis=1)`
  """
  code = "res = torch.sum(input=temp, dim=1)"
  result = rewrite_code(engine, code)
  assert "res = jax.numpy.sum(a=temp, axis=1)" in result


def test_positional_passthrough(engine: PivotRewriter) -> None:
  """
  Test that positional arguments are preserved in order.

  Input:  `torch.div(a, b)`
  Output: `jax.numpy.divide(a, b)`
  """
  code = "res = torch.div(val_1, val_2)"
  result = rewrite_code(engine, code)
  assert "res = jax.numpy.divide(val_1, val_2)" in result


def test_mixed_args_normalization(engine: PivotRewriter) -> None:
  """
  Test mixed positional and keyword arguments.

  Input:  `torch.sum(my_tensor, dim=2)`
  Logic:  my_tensor -> x (pos 0) -> a. dim -> axis.
  Output: `jax.numpy.sum(my_tensor, axis=2)`
  """
  code = "res = torch.sum(my_tensor, dim=2)"
  result = rewrite_code(engine, code)
  assert "res = jax.numpy.sum(my_tensor, axis=2)" in result


def test_unknown_keyword_passthrough(engine: PivotRewriter) -> None:
  """
  Test that extra unknown keyword arguments are preserved (e.g. kwargs).
  """
  code = "res = torch.sum(x, keepdims=True)"
  result = rewrite_code(engine, code)
  # keepdims is not in mock spec, so it passes through without renaming
  assert "res = jax.numpy.sum(x, keepdims=True)" in result


def test_excess_positional_passthrough(engine: PivotRewriter) -> None:
  """
  Test that extra positional arguments are preserved.
  """
  code = "res = torch.div(a, b, c)"
  result = rewrite_code(engine, code)
  assert "res = jax.numpy.divide(a, b, c)" in result


def test_typed_arguments_handling(engine: PivotRewriter) -> None:
  """
  Test that arguments defined as tuples `(name, type)` in `std_args`
  are correctly extracted and mapped.
  """
  # Input uses keyword 'low', which matches 'low' in spec (identity map)
  code = "r = torch.randint(low=0, high=10)"
  result = rewrite_code(engine, code)
  assert "r = jax.random.randint(low=0, high=10)" in result


def test_source_arg_name_precedence(engine: PivotRewriter) -> None:
  """
  Test that if a source keyword matches a standard name directly, it is used.
  (Checking fallback `std_name = k_name` logic).
  """
  # 'x' is the standard name. Source map is {'x': 'input'}.
  # If user uses `torch.sum(x=...)` (invalid in torch but valid python),
  # it should map to 'x' standard arg, and then to 'a' target arg.
  code = "res = torch.sum(x=tensor)"
  result = rewrite_code(engine, code)
  assert "a=tensor" in result


def test_argument_injection(engine: PivotRewriter) -> None:
  """
  Test that arguments defined in `inject_args` are added to the call.

  Input: `torch.normalize(data)`
  Output: `jax.nn.normalize(data, epsilon=1e-05, flag=True)` (Values from mock)
  """
  code = "y = torch.normalize(data)"
  result = rewrite_code(engine, code)

  assert "jax.nn.normalize(data" in result
  assert "epsilon=1e-05" in result
  assert "flag=True" in result

  # Ensure commas handled
  clean = result.replace(" ", "")
  assert ",epsilon=1e-05" in clean
  assert ",flag=True" in clean


def test_argument_value_mapping_strings(engine: PivotRewriter) -> None:
  """
  Test argument value mapping for strings.
  Input: `torch.reduce(x, reduction='mean')`
  Logic: 'reduction' maps to std 'val'. Map for 'val': 'mean' -> 'avg'.
  Target Arg Name: 'mode'.
  Output: `jax.reduce(x, mode='avg')`
  """
  code = "y = torch.reduce(x, reduction='mean')"
  result = rewrite_code(engine, code)

  assert "jax.reduce" in result
  assert "mode='avg'" in result


def test_argument_value_mapping_ints(engine: PivotRewriter) -> None:
  """
  Test argument value mapping for integers.
  Input: `torch.reduce(x, reduction=0)`
  Logic: 0 -> 'none'.
  Output: `jax.reduce(x, mode='none')`
  """
  code = "y = torch.reduce(x, reduction=0)"
  result = rewrite_code(engine, code)

  assert "mode='none'" in result


def test_argument_value_mapping_positional(engine: PivotRewriter) -> None:
  """
  Test argument value mapping for positional args.
  Input: `torch.reduce(x, 'mean')`
  Output: `jax.reduce(x, 'avg')`
  """
  code = "y = torch.reduce(x, 'mean')"
  result = rewrite_code(rewriter=engine, code=code)

  # For positional args, we update the value but don't add keywords unless logic forces it
  # Current NormalizationMixin keeps positional if source was positional
  assert "jax.reduce(x, 'avg')" in result


def test_module_alias_detection(engine: PivotRewriter) -> None:
  """
  Test dynamic module alias detection logic.
  Verify that `torch.method_op(y)` is NOT treated as `method_op(torch, y)`.
  Verify that `var.method_op(y)` is treated as `method_op(var, y)`.
  """
  # Case 1: Framework call (torch defined in Mock)
  # _is_module_alias('torch') should return True
  code_fw = "torch.method_op(y)"
  # BaseRewriter would rewrite this if configured, but here we check internal logic implicitly via output
  # If treated as method, output would be jax.method_op(torch, y) which is wrong.
  # If treated as function, output is jax.method_op(y).
  res_fw = rewrite_code(engine, code_fw)
  clean_fw = res_fw.replace(" ", "")
  assert "(torch," not in clean_fw
  assert "(y)" in clean_fw

  # Case 2: Instance call
  # _is_module_alias('my_obj') should return False (not in framework configs)
  code_inst = "my_obj.method_op(y)"
  # Should inject receiver 'my_obj' as first arg

  # CRITICAL FIX: Ensure 'my_obj.method_op' maps to the definition in reverse index
  # so that the rewriter engages the normalization logic.
  # We grab the definition for 'method_op' first.
  op_def = engine.semantics.data["method_op"]
  engine.semantics._reverse_index["my_obj.method_op"] = ("method_op", op_def)

  res_inst = rewrite_code(engine, code_inst)
  clean_inst = res_inst.replace(" ", "")
  # Assertion Fixed: verify my_obj is present as arg
  # e.g. jax.method_op(my_obj, y)
  assert "(my_obj,y)" in clean_inst

  # Case 3: Experimental Framework (defined in Mock configs)
  # Should behave like torch
  code_exp = "exp.net.method_op(y)"
  res_exp = rewrite_code(engine, code_exp)
  # Should not inject exp.net as argument

  # Temporarily injecting into rewriter instance map needed here too because
  # 'exp.net.method_op' isn't standardly mapped in init
  engine.semantics._reverse_index["exp.net.method_op"] = ("method_op", op_def)

  res_exp = rewrite_code(engine, code_exp)
  clean_exp = res_exp.replace(" ", "")
  # Should be jax.method_op(y) NOT jax.method_op(exp.net, y)
  assert "(exp.net," not in clean_exp
  assert "(y)" in clean_exp
