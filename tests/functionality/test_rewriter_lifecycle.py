"""Test suite for the Rewriter Lifecycle module."""

import typing

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
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Any] = {}
    self._known_rng_methods: set[str] = set()
    self._validation_status: dict[str, typing.Any] = {}
    self._providers: dict[str, typing.Any] = {}
    self._source_registry: dict[str, typing.Any] = {}
    self._inject("abs", "torch.abs", "jax.numpy.abs")
    self._inject("float32", "torch.float32", "jax.numpy.float32")
    self._inject("future_op", "torch.future", "jax.future", min_v="9.0.0")
    self._inject("legacy_op", "torch.legacy", "jax.legacy", max_v="0.0.1")
    self._inject("old_scatter", "torch.old_scatter", "jax.scatter", deprecated=True, replaced_by="Scatter")
    self._inject("unsafe_op", "torch.unsafe", "jax.unsafe", deprecated=True)
    self.framework_configs: dict[str, typing.Any] = {
      "torch": {
        "traits": {
          "lifecycle_strip_methods": ["to", "cpu", "cuda", "detach"],
          "lifecycle_warn_methods": ["eval", "train"],
        }
      },
      "jax": {"version": "1.0.0"},
    }

  def get_framework_config(self, framework: str) -> typing.Any:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return True

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get_definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)

  def _inject(
    self,
    name: str,
    s_api: str,
    t_api: str,
    min_v: typing.Optional[str] = None,
    max_v: typing.Optional[str] = None,
    deprecated: bool = False,
    replaced_by: typing.Optional[str] = None,
  ) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
        min_v: Test argument or fixture.
        max_v: Test argument or fixture.
        deprecated: Test argument or fixture.
        replaced_by: Test argument or fixture.
    """
    tgt_var: dict[str, typing.Any] = {"api": t_api}
    if min_v:
      tgt_var["min_version"] = min_v
    if max_v:
      tgt_var["max_version"] = max_v
    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": tgt_var}, "std_args": ["x"]}
    if deprecated:
      self.data[name]["deprecated"] = True
    if replaced_by:
      self.data[name]["replaced_by"] = replaced_by
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockSemantics()
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
  try:
    new_tree: typing.Any = rewriter.convert(tree)
    return typing.cast(str, new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewriter crashed: {e}")


def test_strip_to_call(rewriter: TestRewriter) -> None:
  """Verifies the behavior of strip to call.

  Args:
      rewriter: Test argument or fixture.
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
      rewriter: Test argument or fixture.
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
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.future(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.future(x)" in result
  assert EscapeHatch.START_MARKER in result
  assert "Target jax@1.0.0 is older than required 9.0.0" in result


def test_deprecation_warning(rewriter: TestRewriter) -> None:
  """Verifies the behavior of deprecation warning.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "y = torch.unsafe(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.unsafe(x)" in result
  assert "Usage of deprecated operation 'unsafe_op'" in result
