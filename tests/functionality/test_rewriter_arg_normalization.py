"""
Tests for Argument Normalization Logic in TestRewriter.

Verifies that the Rewriter correctly pivots argument names and positions
based on the abstract specification.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockArgSemantics(SemanticsManager):
  """
  Mock Data for Argument Normalization tests.
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
    # Source: sum(input, dim) -> Standard: sum(x, axis) -> Target: sum(a, axis)
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
    self._inject_op(
      op_name="normalize",
      std_args=["x"],
      variants={
        "torch": {"api": "torch.normalize"},
        "jax": {"api": "jax.nn.normalize", "inject_args": {"epsilon": 1e-5, "flag": True}},
      },
    )

    # 5. Value Mapping Scenario ('reduce')
    self._inject_op(
      op_name="reduce",
      std_args=["x", "val"],
      variants={
        "torch": {"api": "torch.reduce", "args": {"val": "reduction"}},
        "jax": {"api": "jax.reduce", "args": {"val": "mode"}, "arg_values": {"val": {"mean": "'avg'", "0": "'none'"}}},
      },
    )

    # 6. Method Style Test ('method_op')
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
def engine() -> TestRewriter:
  """Returns a Rewriter for Torch -> JAX arg testing."""
  semantics = MockArgSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(semantics, config)


def rewrite_code(rewriter: TestRewriter, code: str) -> str:
  """Parses code, applies rewriter, and returns generated code."""
  tree = cst.parse_module(code)
  return rewriter.convert(tree).code


def test_keyword_translation(engine: TestRewriter) -> None:
  """
  Input:  `torch.sum(input=z, dim=1)`
  Logic:  input -> x -> a, dim -> axis -> axis.
  Output: `jax.numpy.sum(a=z, axis=1)`
  """
  code = "res = torch.sum(input=temp, dim=1)"
  result = rewrite_code(engine, code)
  assert "res = jax.numpy.sum(a=temp, axis=1)" in result


def test_positional_passthrough(engine: TestRewriter) -> None:
  code = "res = torch.div(val_1, val_2)"
  result = rewrite_code(engine, code)
  assert "res = jax.numpy.divide(val_1, val_2)" in result


def test_mixed_args_normalization(engine: TestRewriter) -> None:
  code = "res = torch.sum(my_tensor, dim=2)"
  result = rewrite_code(engine, code)
  assert "res = jax.numpy.sum(my_tensor, axis=2)" in result


def test_unknown_keyword_passthrough(engine: TestRewriter) -> None:
  code = "res = torch.sum(x, keepdims=True)"
  result = rewrite_code(engine, code)
  assert "res = jax.numpy.sum(x, keepdims=True)" in result


def test_typed_arguments_handling(engine: TestRewriter) -> None:
  code = "r = torch.randint(low=0, high=10)"
  result = rewrite_code(engine, code)
  assert "r = jax.random.randint(low=0, high=10)" in result


def test_argument_injection(engine: TestRewriter) -> None:
  code = "y = torch.normalize(data)"
  result = rewrite_code(engine, code)

  assert "jax.nn.normalize(data" in result
  assert "epsilon=1e-05" in result
  assert "flag=True" in result


def test_argument_value_mapping_strings(engine: TestRewriter) -> None:
  code = "y = torch.reduce(x, reduction='mean')"
  result = rewrite_code(engine, code)

  assert "jax.reduce" in result
  assert "mode='avg'" in result


def test_module_alias_detection(engine: TestRewriter) -> None:
  """
  Test dynamic module alias detection logic.
  """
  # Case 1: Framework call
  code_fw = "torch.method_op(y)"
  res_fw = rewrite_code(engine, code_fw)
  clean_fw = res_fw.replace(" ", "")
  assert "(torch," not in clean_fw
  assert "(y)" in clean_fw

  # Case 2: Instance call
  code_inst = "my_obj.method_op(y)"
  # Ensure reverse lookup
  op_def = engine.semantics.data["method_op"]
  engine.semantics._reverse_index["my_obj.method_op"] = ("method_op", op_def)

  res_inst = rewrite_code(engine, code_inst)
  clean_inst = res_inst.replace(" ", "")
  assert "(my_obj,y)" in clean_inst
