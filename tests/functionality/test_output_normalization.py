"""Test suite for the Output Normalization module."""

import typing
from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockOutputSemantics(SemanticsManager):
  """Test helper class MockOutputSemantics."""

  def __init__(self) -> None:
    """Initializes the MockOutputSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
    self.framework_configs = {}
    self._inject("split_vals", "torch.split", "jax.numpy.split", select_index=0)

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: str, select_index: int | None = None) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
        select_index: Test argument or fixture.
    """
    variants = {"torch": {"api": s_api}, "jax": {"api": t_api}}
    target_var = variants["jax"]
    if select_index is not None:
      getattr(getattr({"v": target_var}, "__getitem__")("v"), "__setitem__")(
        "output_select_index", getattr({"v": select_index}, "__getitem__")("v")
      )
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "variants": getattr({"v": variants}, "__getitem__")("v"),
            "std_args": getattr(
              {"v": [getattr({"v": "x"}, "__getitem__")("v"), getattr({"v": "y"}, "__getitem__")("v")]}, "__getitem__"
            )("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    self._reverse_index[s_api] = (name, self.data[name])

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
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

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Mock implementation of get import map.

    Args:
        target_framework: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockOutputSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
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
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


def test_structured_index_wrapping(rewriter: TestRewriter) -> None:
  """Verifies the behavior of structured index wrapping.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "res = torch.split(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.split(x)[0]" in result
  assert "lambda" not in result
