"""Test suite for the Rewriter Decorators module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OpType
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockDecoratorSemantics(SemanticsManager):
  """Test helper class MockDecoratorSemantics."""

  def __init__(self) -> None:
    """Initializes the MockDecoratorSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {}
    self._inject("jit", "torch.jit.script", "jax.jit")
    self._inject("inference_mode", "torch.inference_mode", None)
    self._inject("compile", "torch.compile", "jax.jit")

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: str | None) -> None:
    """Mock implementation of  inject.

    Args:
        name: Identifier or operation name string.
        s_api: The s api (str) for inject.
        t_api: The t api (str | None) for inject.
    """
    variants = {"torch": {"api": s_api}}
    if t_api is None:
      getattr(getattr({"v": variants}, "__getitem__")("v"), "__setitem__")(
        "jax", getattr({"v": None}, "__getitem__")("v")
      )
    else:
      variants["jax"] = {"api": t_api}
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "op_type": getattr({"v": getattr({"v": OpType}, "__getitem__")("v").DECORATOR}, "__getitem__")("v"),
            "variants": getattr({"v": variants}, "__getitem__")("v"),
            "std_args": getattr({"v": [getattr({"v": "fn"}, "__getitem__")("v")]}, "__getitem__")("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    self._reverse_index[s_api] = (name, self.data[name])

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get_definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve_variant.

    Args:
        abstract_id: The abstract id (str) for resolve variant.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v").get},
                          "__getitem__",
                        )("v")(getattr({"v": abstract_id}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
                      },
                      "__getitem__",
                    )("v").get
                  },
                  "__getitem__",
                )("v")(getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
              },
              "__getitem__",
            )("v").get
          },
          "__getitem__",
        )("v")(getattr({"v": fw}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (TestRewriter).
  """
  semantics = MockDecoratorSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `str` from rewrite.
  """
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return str(new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewriter failed: {e}")


def test_decorator_renaming(rewriter: TestRewriter) -> None:
  """Verifies the behavior of decorator renaming.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\n@torch.jit.script\ndef func(x):\n    return x\n"
  result: str = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.jit.script" not in result


def test_decorator_removal(rewriter: TestRewriter) -> None:
  """Verifies the behavior of decorator removal.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\n@torch.inference_mode\ndef func(x):\n    return x\n"
  result: str = rewrite(rewriter, code)
  assert "@torch.inference_mode" not in result
  assert "def func(x):" in result


def test_call_decorator_renaming(rewriter: TestRewriter) -> None:
  """Verifies the behavior of call decorator renaming.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\n@torch.compile(fullgraph=True)\ndef func(x):\n    pass\n"
  result: str = rewrite(rewriter, code)
  assert "@jax.jit(fullgraph=True)" in result
  assert "torch.compile" not in result


def test_multiple_decorators_mixed(rewriter: TestRewriter) -> None:
  """Verifies the behavior of multiple decorators mixed.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\n@torch.jit.script\n@torch.inference_mode\ndef f():\n    pass\n"
  result: str = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.inference_mode" not in result
  assert "def f():" in result
