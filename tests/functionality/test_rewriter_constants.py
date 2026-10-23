"""Test suite for the Rewriter Constants module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {}
    self._inject_const("float32", {"torch": "torch.float32", "jax": "jax.numpy.float32"})
    self._inject_func("abs", {"torch": "torch.abs", "jax": "jax.numpy.abs"})

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(framework, {})

  def _inject_const(self, name: str, mapping: dict[str, str]) -> None:
    """Mock implementation of  inject const.

    Args:
        name: Identifier or operation name string.
        mapping: The mapping (dict[str, str]) for inject const.
    """
    self.data[name] = {"variants": {}}
    for fw, api in mapping.items():
      getattr(
        getattr(
          {
            "v": getattr(
              {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v")[name]},
              "__getitem__",
            )("v")["variants"]
          },
          "__getitem__",
        )("v"),
        "__setitem__",
      )(fw, getattr({"v": {"api": getattr({"v": api}, "__getitem__")("v")}}, "__getitem__")("v"))
      self._reverse_index[api] = (name, self.data[name])

  def _inject_func(self, name: str, mapping: dict[str, str]) -> None:
    """Mock implementation of  inject function.

    Args:
        name: Identifier or operation name string.
        mapping: The mapping (dict[str, str]) for inject func.
    """
    self.data[name] = {"variants": {}, "std_args": ["x"]}
    for fw, api in mapping.items():
      getattr(
        getattr(
          {
            "v": getattr(
              {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v")[name]},
              "__getitem__",
            )("v")["variants"]
          },
          "__getitem__",
        )("v"),
        "__setitem__",
      )(fw, getattr({"v": {"api": getattr({"v": api}, "__getitem__")("v")}}, "__getitem__")("v"))
      self._reverse_index[api] = (name, self.data[name])

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
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(MockSemantics(), config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `str` from rewrite.
  """
  tree = cst.parse_module(code)
  return str(rewriter.convert(tree).code)


def test_constant_rewrite_assignment(rewriter: TestRewriter) -> None:
  """Verifies the behavior of constant rewrite assignment.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "x = torch.float32"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res
  assert "torch.float32" not in res


def test_constant_rewrite_argument(rewriter: TestRewriter) -> None:
  """Verifies the behavior of constant rewrite argument.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = init(dtype=torch.float32)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.float32" in res


def test_function_call_rewrite(rewriter: TestRewriter) -> None:
  """Verifies the behavior of function call rewrite.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.abs(x)"
  res: str = rewrite(rewriter, code)
  assert "jax.numpy.abs(x)" in res
  assert "torch.abs" not in res
