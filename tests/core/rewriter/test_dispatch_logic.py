"""Test suite for the Dispatch Logic module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import LogicOp, Rule
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockDispatchSemantics(SemanticsManager):
  """Test helper class MockDispatchSemantics."""

  def __init__(self) -> None:
    """Initializes the MockDispatchSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {}
    resize_def = {
      "std_args": ["image", "dummy", "mode"],
      "variants": {
        "torch": {"api": "torch.resize", "args": {}},
        "jax": {
          "api": "jax.image.resize",
          "args": {},
          "dispatch_rules": [
            Rule(if_arg="mode", op=LogicOp.EQ, val="nearest", use_api="jax.image.resize_nearest"),
            Rule(if_arg="mode", op=LogicOp.IN, val=["bilinear", "bicubic"], use_api="jax.image.resize_bi"),
          ],
        },
      },
    }
    getattr(self.data, "__setitem__")("resize", resize_def)
    getattr(self._reverse_index, "__setitem__")(
      "torch.resize",
      ("resize", resize_def),
    )
    clamp_def = {
      "std_args": ["x", "limit"],
      "variants": {
        "torch": {"api": "torch.clamp"},
        "jax": {
          "api": "jnp.clip",
          "dispatch_rules": [Rule(if_arg="limit", op=LogicOp.GT, val=100, use_api="jnp.heavy_clip")],
        },
      },
    }
    getattr(self.data, "__setitem__")("clamp", clamp_def)
    getattr(self._reverse_index, "__setitem__")(
      "torch.clamp",
      ("clamp", clamp_def),
    )
    process_def = {
      "std_args": ["data"],
      "variants": {
        "torch": {"api": "torch.process"},
        "jax": {
          "api": "jax.single_process",
          "dispatch_rules": [
            Rule(if_arg="data", op=LogicOp.IS_TYPE, val="list", use_api="jax.batch_process"),
            Rule(if_arg="data", op=LogicOp.IS_TYPE, val="int", use_api="jax.int_process"),
          ],
        },
      },
    }
    getattr(self.data, "__setitem__")("process", process_def)
    getattr(self._reverse_index, "__setitem__")(
      "torch.process",
      ("process", process_def),
    )

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Tuple of abstract operation ID and definition metadata, or None.
    """
    if name.endswith("resize"):
      return ("resize", self.data["resize"])
    if name.endswith("clamp"):
      return ("clamp", self.data["clamp"])
    if name.endswith("process"):
      return ("process", self.data["process"])
    return self._reverse_index.get(name)

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (TestRewriter).
  """
  semantics = MockDispatchSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Transpiled source code string or rewritten CST module.
  """
  tree = cst.parse_module(code)
  return str(rewriter.convert(tree).code)


def test_dispatch_equality_string(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch equality string.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.resize(x, None, mode='nearest')"
  res: str = rewrite(rewriter, code)
  assert "jax.image.resize_nearest" in res
  assert "mode='nearest'" in res


def test_dispatch_fallback_default(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch fallback default.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.resize(x, None, mode='linear')"
  res: str = rewrite(rewriter, code)
  assert "jax.image.resize(" in res


def test_dispatch_in_list(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch in list.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.resize(x, None, mode='bicubic')"
  res: str = rewrite(rewriter, code)
  assert "jax.image.resize_bi" in res


def test_dispatch_positional_extraction(rewriter: TestRewriter) -> None:
  """Verify dispatch positional extraction behavior.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.resize(x, None, 'nearest')"
  res: str = rewrite(rewriter, code)
  assert "jax.image.resize_nearest" in res


def test_dispatch_numeric_gt(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch numeric gt.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.clamp(x, 150)"
  res: str = rewrite(rewriter, code)
  assert "jnp.heavy_clip" in res


def test_dispatch_numeric_method_call(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch numeric method call.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = x.clamp(50)"
  res: str = rewrite(rewriter, code)
  assert "jnp.clip" in res
  code2: str = "y = x.clamp(150)"
  res2: str = rewrite(rewriter, code2)
  assert "jnp.heavy_clip" in res2


def test_dispatch_is_type_list(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch is type list.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.process([1, 2])"
  res: str = rewrite(rewriter, code)
  assert "jax.batch_process" in res


def test_dispatch_is_type_int(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch is type integer.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.process(5)"
  res: str = rewrite(rewriter, code)
  assert "jax.int_process" in res


def test_dispatch_is_type_fallback(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch is type fallback.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.process(x)"
  res: str = rewrite(rewriter, code)
  assert "jax.single_process" in res
