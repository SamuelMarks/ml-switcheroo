"""Test suite for the Output Casting module."""

import typing
from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockCastSemantics(SemanticsManager):
  """Test helper class MockCastSemantics."""

  def __init__(self) -> None:
    """Initializes the MockCastSemantics instance."""
    self.data = {}
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.test_templates = {}
    self._known_rng_methods: set[str] = set()
    self._inject("ArgMax", "torch.argmax", "jax.numpy.argmax", output_cast="jnp.int64")
    self._inject("Normalize", "torch.simple_op", "jax.op", output_cast="jnp.float32")

  def get_all_rng_methods(self) -> set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Computed test result or fixture value.
    """
    return set()

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return {}

  def _inject(self, name: str, s_api: str, t_api: str, output_cast: str | None = None) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
        output_cast: Test argument or fixture.
    """
    t_def = {"api": t_api}
    if output_cast:
      t_def["output_cast"] = output_cast
    variants = {"torch": {"api": s_api}, "jax": t_def}
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "variants": getattr({"v": variants}, "__getitem__")("v"),
            "std_args": getattr({"v": [getattr({"v": "x"}, "__getitem__")("v")]}, "__getitem__")("v"),
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


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockCastSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite(rewriter: PivotRewriter, code: str) -> str:
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


def test_output_cast_injection(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of output cast injection.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.argmax(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.argmax(x)" in result
  assert ".astype(jnp.int64)" in result


def test_output_cast_float_conversion(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of output cast float conversion.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "z = torch.simple_op(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.op(x)" in result
  assert ".astype(jnp.float32)" in result
