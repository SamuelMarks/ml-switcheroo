"""Test suite for the Rewriter Arg Normalization module."""

import pytest
import libcst as cst
import typing
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockArgSemantics(SemanticsManager):
  """Mock Arg Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockArgSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self.import_data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self.framework_configs: dict[str, typing.Any] = {
      "torch": {"alias": {"module": "torch", "name": "torch"}},
      "jax": {"alias": {"module": "jax.numpy", "name": "jnp"}},
      "experimental_fw": {"alias": {"module": "exp.net", "name": "exp"}},
    }
    self._inject_op(
      op_name="sum",
      std_args=["x", "axis"],
      variants={
        "torch": {"api": "torch.sum", "args": {"x": "input", "axis": "dim"}},
        "jax": {"api": "jax.numpy.sum", "args": {"x": "a", "axis": "axis"}},
      },
    )
    self._inject_op(
      op_name="div",
      std_args=["x", "y"],
      variants={"torch": {"api": "torch.div"}, "jax": {"api": "jax.numpy.divide"}},
    )
    self._inject_op(
      op_name="randint",
      std_args=[{"name": "low", "type": "int"}, {"name": "high", "type": "int"}],
      variants={"torch": {"api": "torch.randint"}, "jax": {"api": "jax.random.randint"}},
    )
    self._inject_op(
      op_name="normalize",
      std_args=["x"],
      variants={
        "torch": {"api": "torch.normalize"},
        "jax": {"api": "jax.nn.normalize", "inject_args": {"epsilon": "1e-05", "flag": "True"}},
      },
    )
    self._inject_op(
      op_name="reduce",
      std_args=["x", "val"],
      variants={
        "torch": {"api": "torch.reduce", "args": {"val": "reduction"}},
        "jax": {"api": "jax.reduce", "args": {"val": "mode"}, "arg_values": {"val": {"mean": "'avg'", "0": "'none'"}}},
      },
    )
    self._inject_op(
      op_name="method_op",
      std_args=["x", "y"],
      variants={"torch": {"api": "torch.method_op"}, "jax": {"api": "jax.method_op"}},
    )

  def _inject_op(self, op_name: str, std_args: list[typing.Any], variants: dict[str, typing.Any]) -> None:
    """Mock implementation of  inject op."""
    self.data[op_name] = {"std_args": std_args, "variants": variants}
    for _, details in variants.items():
      if "api" in details:
        self._reverse_index[details["api"]] = (op_name, self.data[op_name])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, aid: str, fw: str) -> typing.Any:
    """Mock resolve variant."""
    return self.data.get(aid, {}).get("variants", {}).get(fw)

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified."""
    return True


@pytest.fixture
def engine() -> TestRewriter:
  """Provides a mock engine for testing."""
  semantics = MockArgSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(semantics, config)


def rewrite_code(rewriter: TestRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  return typing.cast(str, rewriter.convert(tree).code)


def test_keyword_translation(engine: TestRewriter) -> None:
  """Verifies the behavior of keyword translation."""
  code: str = "res = torch.sum(input=temp, dim=1)"
  result: str = rewrite_code(engine, code)
  assert "res = jax.numpy.sum(a=temp, axis=1)" in result


def test_positional_passthrough(engine: TestRewriter) -> None:
  """Verifies the behavior of positional passthrough."""
  code: str = "res = torch.div(val_1, val_2)"
  result: str = rewrite_code(engine, code)
  assert "res = jax.numpy.divide(val_1, val_2)" in result


def test_mixed_args_normalization(engine: TestRewriter) -> None:
  """Verifies the behavior of mixed arguments normalization."""
  code: str = "res = torch.sum(my_tensor, dim=2)"
  result: str = rewrite_code(engine, code)
  assert "res = jax.numpy.sum(my_tensor, axis=2)" in result


def test_unknown_keyword_passthrough(engine: TestRewriter) -> None:
  """Verifies the behavior of unknown keyword passthrough."""
  code: str = "res = torch.sum(x, keepdims=True)"
  result: str = rewrite_code(engine, code)
  assert "res = jax.numpy.sum(x, keepdims=True)" in result


def test_typed_arguments_handling(engine: TestRewriter) -> None:
  """Verifies the behavior of typed arguments handling."""
  code: str = "r = torch.randint(low=0, high=10)"
  result: str = rewrite_code(engine, code)
  assert "r = jax.random.randint(low=0, high=10)" in result


def test_argument_injection(engine: TestRewriter) -> None:
  """Verifies the behavior of argument injection."""
  code: str = "y = torch.normalize(data)"
  result: str = rewrite_code(engine, code)
  assert "jax.nn.normalize(data" in result
  assert "epsilon=1e-05" in result
  assert "flag=True" in result


def test_argument_value_mapping_strings(engine: TestRewriter) -> None:
  """Verifies the behavior of argument value mapping strings."""
  code: str = "y = torch.reduce(x, reduction='mean')"
  result: str = rewrite_code(engine, code)
  assert "jax.reduce" in result
  assert "mode='avg'" in result


def test_module_alias_detection(engine: TestRewriter) -> None:
  """Verifies the behavior of module alias detection."""
  code_fw: str = "torch.method_op(y)"
  res_fw: str = rewrite_code(engine, code_fw)
  clean_fw: str = res_fw.replace(" ", "")
  assert "(torch," not in clean_fw
  assert "(y)" in clean_fw
  code_inst: str = "my_obj.method_op(y)"
  op_def: typing.Any = engine.semantics.data["method_op"]
  engine.semantics._reverse_index["my_obj.method_op"] = ("method_op", op_def)  # type: ignore
  res_inst: str = rewrite_code(engine, code_inst)
  clean_inst: str = res_inst.replace(" ", "")
  assert "(my_obj,y)" in clean_inst
