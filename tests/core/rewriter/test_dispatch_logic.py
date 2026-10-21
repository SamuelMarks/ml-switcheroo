"""Test suite for the Dispatch Logic module."""

import typing
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
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
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
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      "resize", getattr({"v": resize_def}, "__getitem__")("v")
    )
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v")._reverse_index}, "__getitem__")("v"), "__setitem__")(
      "torch.resize",
      getattr(
        {"v": (getattr({"v": "resize"}, "__getitem__")("v"), getattr({"v": resize_def}, "__getitem__")("v"))},
        "__getitem__",
      )("v"),
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
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      "clamp", getattr({"v": clamp_def}, "__getitem__")("v")
    )
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v")._reverse_index}, "__getitem__")("v"), "__setitem__")(
      "torch.clamp",
      getattr(
        {"v": (getattr({"v": "clamp"}, "__getitem__")("v"), getattr({"v": clamp_def}, "__getitem__")("v"))}, "__getitem__"
      )("v"),
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
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      "process", getattr({"v": process_def}, "__getitem__")("v")
    )
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v")._reverse_index}, "__getitem__")("v"), "__setitem__")(
      "torch.process",
      getattr(
        {"v": (getattr({"v": "process"}, "__getitem__")("v"), getattr({"v": process_def}, "__getitem__")("v"))},
        "__getitem__",
      )("v"),
    )

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
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
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockDispatchSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Test argument or fixture.
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  tree = cst.parse_module(code)
  return str(rewriter.convert(tree).code)


def test_dispatch_equality_string(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch equality string.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.resize(x, None, mode='nearest')"
  res: str = rewrite(rewriter, code)
  assert "jax.image.resize_nearest" in res
  assert "mode='nearest'" in res


def test_dispatch_fallback_default(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch fallback default.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.resize(x, None, mode='linear')"
  res: str = rewrite(rewriter, code)
  assert "jax.image.resize(" in res


def test_dispatch_in_list(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch in list.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.resize(x, None, mode='bicubic')"
  res: str = rewrite(rewriter, code)
  assert "jax.image.resize_bi" in res


def test_dispatch_positional_extraction(rewriter: TestRewriter) -> None:
  """Verify dispatch positional extraction behavior.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.resize(x, None, 'nearest')"
  res: str = rewrite(rewriter, code)
  assert "jax.image.resize_nearest" in res


def test_dispatch_numeric_gt(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch numeric gt.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.clamp(x, 150)"
  res: str = rewrite(rewriter, code)
  assert "jnp.heavy_clip" in res


def test_dispatch_numeric_method_call(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch numeric method call.

  Args:
      rewriter: Test argument or fixture.
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
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.process([1, 2])"
  res: str = rewrite(rewriter, code)
  assert "jax.batch_process" in res


def test_dispatch_is_type_int(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch is type integer.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.process(5)"
  res: str = rewrite(rewriter, code)
  assert "jax.int_process" in res


def test_dispatch_is_type_fallback(rewriter: TestRewriter) -> None:
  """Verifies the behavior of dispatch is type fallback.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.process(x)"
  res: str = rewrite(rewriter, code)
  assert "jax.single_process" in res
