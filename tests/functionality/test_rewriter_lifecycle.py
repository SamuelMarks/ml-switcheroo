"""Test suite for the Rewriter Lifecycle module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
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
    self._known_rng_methods: set[str] = set()
    self._validation_status = {}
    self._providers = {}
    self._source_registry = {}
    self._inject("abs", "torch.abs", "jax.numpy.abs")
    self._inject("float32", "torch.float32", "jax.numpy.float32")
    self._inject("future_op", "torch.future", "jax.future", min_v="9.0.0")
    self._inject("legacy_op", "torch.legacy", "jax.legacy", max_v="0.0.1")
    self.framework_configs = {
      "torch": {
        "traits": {
          "lifecycle_strip_methods": ["to", "cpu", "cuda", "detach"],
          "lifecycle_warn_methods": ["eval", "train"],
        }
      },
      "jax": {"version": "1.0.0"},
    }

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(framework, {})

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: The  id (str) for is verified.

    Returns:
        Resulting `bool` from is verified.
    """
    return True

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get_definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve variant.

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

  def _inject(
    self,
    name: str,
    s_api: str,
    t_api: str,
    min_v: str | None = None,
    max_v: str | None = None,
  ) -> None:
    """Mock implementation of  inject.

    Args:
        name: Identifier or operation name string.
        s_api: The s api (str) for inject.
        t_api: The t api (str) for inject.
        min_v: The min v (str | None) for inject.
        max_v: The max v (str | None) for inject.
    """
    tgt_var = {"api": t_api}
    if min_v:
      tgt_var["min_version"] = min_v
    if max_v:
      tgt_var["max_version"] = max_v
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "variants": getattr(
              {
                "v": {
                  "torch": getattr({"v": {"api": getattr({"v": s_api}, "__getitem__")("v")}}, "__getitem__")("v"),
                  "jax": getattr({"v": tgt_var}, "__getitem__")("v"),
                }
              },
              "__getitem__",
            )("v"),
            "std_args": getattr({"v": [getattr({"v": "x"}, "__getitem__")("v")]}, "__getitem__")("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (TestRewriter).
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
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
    pytest.fail(f"Rewriter crashed: {e}")


def test_strip_to_call(rewriter: TestRewriter) -> None:
  """Verifies the behavior of strip to call.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "x = tensor.to(device)"
  result: str = rewrite(rewriter, code)
  assert "x = tensor" in result
  is_to_present: bool = any((".to(" in line and (not line.strip().startswith("#")) for line in result.splitlines()))
  assert not is_to_present
  assert EscapeHatch.START_MARKER in result
  assert "Stripped framework-specific lifecycle method '.to()'" in result


def test_warn_on_eval_train(rewriter: TestRewriter) -> None:
  """Verifies the behavior of warn on eval train.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "model.eval()"
  result: str = rewrite(rewriter, code)
  is_eval: bool = any(("model.eval" in line and (not line.strip().startswith("#")) for line in result.splitlines()))
  assert not is_eval
  assert EscapeHatch.START_MARKER in result
  assert "Ignored model state method '.eval()'" in result


def test_version_constraint_check_min(rewriter: TestRewriter) -> None:
  """Verifies the behavior of version constraint check min.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.future(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.future(x)" in result
  assert EscapeHatch.START_MARKER in result
  assert "Target jax@1.0.0 is older than required 9.0.0" in result


def test_version_constraint_check_max(rewriter: TestRewriter) -> None:
  """Verifies the behavior of version constraint check max.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.legacy(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.legacy(x)" in result
  assert EscapeHatch.START_MARKER in result
  assert "Target jax@1.0.0 exceeds max supported 0.0.1" in result
