"""Test suite for the Stack Wiring module."""

import typing
from pydantic import JsonValue

import pytest

from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


class MockAdapter(JAXStackMixin):
  """Test helper class MockAdapter."""

  def apply_wiring(self, snapshot: dict[str, JsonValue]) -> None:
    """Mock implementation of apply wiring.

    Args:
        snapshot: Test argument or fixture.
    """
    self._apply_stack_wiring(snapshot)  # type: ignore[arg-type]


@pytest.fixture
def adapter() -> MockAdapter:
  """Execute adapter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return MockAdapter()


def test_core_math_mappings(adapter: MockAdapter) -> None:
  """Verifies the behavior of core math mappings.

  Args:
      adapter: Test argument or fixture.
  """
  snapshot: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}
  adapter.apply_wiring(snapshot)  # type: ignore[arg-type]
  mappings = snapshot["mappings"]
  assert "Adam" in mappings  # type: ignore[operator]
  assert mappings["Adam"]["api"] == "optax.adam"  # type: ignore[index]
  assert "step" in mappings  # type: ignore[operator]
  assert mappings["step"]["requires_plugin"] == "optimizer_step"  # type: ignore[index]


def test_optax_autowiring(adapter: MockAdapter) -> None:
  """Verifies the behavior of optax autowiring.

  Args:
      adapter: Test argument or fixture.
  """
  snapshot: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}
  adapter.apply_wiring(snapshot)  # type: ignore[arg-type]
  mappings = snapshot["mappings"]
  assert "Adam" in mappings  # type: ignore[operator]
  assert mappings["Adam"]["api"] == "optax.adam"  # type: ignore[index]
  assert mappings["Adam"]["requires_plugin"] == "optimizer_constructor"  # type: ignore[index]
  assert "step" in mappings  # type: ignore[operator]
  assert mappings["step"]["requires_plugin"] == "optimizer_step"  # type: ignore[index]


def test_io_serialization_wiring(adapter: MockAdapter) -> None:
  """Verifies the behavior of I/O serialization wiring.

  Args:
      adapter: Test argument or fixture.
  """
  imports: list[str] = adapter.get_serialization_imports()
  assert "import orbax.checkpoint" in imports[0]
  save_code: str = adapter.get_serialization_syntax(op="save", file_arg="'./ckpt'", object_arg="state")
  assert "orbax.checkpoint.PyTreeCheckpointer().save" in save_code
  assert "directory='./ckpt'" in save_code
  assert "item=state" in save_code
  load_code: str = adapter.get_serialization_syntax(op="load", file_arg="'./ckpt'")
  assert "orbax.checkpoint.PyTreeCheckpointer().restore" in load_code
  assert "('./ckpt')" in load_code


def test_control_flow_templates(adapter: MockAdapter) -> None:
  """Verifies the behavior of control flow templates.

  Args:
      adapter: Test argument or fixture.
  """
  snapshot: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}
  adapter.apply_wiring(snapshot)  # type: ignore[arg-type]
  templates = snapshot.get("templates", {})
  assert "fori_loop" in templates  # type: ignore[operator]
  assert "jax.lax.fori_loop" in templates["fori_loop"]  # type: ignore[index]
  assert "scan" in templates  # type: ignore[operator]
  assert "jax.lax.scan" in templates["scan"]  # type: ignore[index]


def test_device_syntax_generation(adapter: MockAdapter) -> None:
  """Verifies the behavior of device syntax generation.

  Args:
      adapter: Test argument or fixture.
  """
  code_cuda: str = adapter.get_device_syntax("'cuda'")
  assert "jax.devices('gpu')[0]" == code_cuda
  code_cpu: str = adapter.get_device_syntax("'cpu'")
  assert "jax.devices('cpu')[0]" == code_cpu
  code_idx: str = adapter.get_device_syntax("'cuda'", device_index="1")
  assert "jax.devices('gpu')[1]" == code_idx
  code_var: str = adapter.get_device_syntax("my_device")
  assert "jax.devices(my_device)[0]" == code_var
