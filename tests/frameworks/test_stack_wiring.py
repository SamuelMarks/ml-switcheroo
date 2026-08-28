"""Test suite for the Stack Wiring module."""

import pytest
import typing
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


class MockAdapter(JAXStackMixin):
  """Mock Adapter class for testing purposes."""

  def apply_wiring(self, snapshot: dict[str, typing.Any]) -> None:
    """Mock implementation of apply wiring."""
    self._apply_stack_wiring(snapshot)


@pytest.fixture
def adapter() -> MockAdapter:
  """Provides a mock adapter for testing."""
  return MockAdapter()


def test_core_math_mappings(adapter: MockAdapter) -> None:
  """Verifies the behavior of core math mappings."""
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  mappings: typing.Any = snapshot["mappings"]
  assert "Adam" in mappings
  assert mappings["Adam"]["api"] == "optax.adam"
  assert "step" in mappings
  assert mappings["step"]["requires_plugin"] == "optimizer_step"


def test_optax_autowiring(adapter: MockAdapter) -> None:
  """Verifies the behavior of optax autowiring."""
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  mappings: typing.Any = snapshot["mappings"]
  assert "Adam" in mappings
  assert mappings["Adam"]["api"] == "optax.adam"
  assert mappings["Adam"]["requires_plugin"] == "optimizer_constructor"
  assert "step" in mappings
  assert mappings["step"]["requires_plugin"] == "optimizer_step"


def test_io_serialization_wiring(adapter: MockAdapter) -> None:
  """Verifies the behavior of I/O serialization wiring."""
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
  """Verifies the behavior of control flow templates."""
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  templates: dict[str, typing.Any] = snapshot.get("templates", {})
  assert "fori_loop" in templates
  assert "jax.lax.fori_loop" in templates["fori_loop"]
  assert "scan" in templates
  assert "jax.lax.scan" in templates["scan"]


def test_device_syntax_generation(adapter: MockAdapter) -> None:
  """Verifies the behavior of device syntax generation."""
  code_cuda: str = adapter.get_device_syntax("'cuda'")
  assert "jax.devices('gpu')[0]" == code_cuda
  code_cpu: str = adapter.get_device_syntax("'cpu'")
  assert "jax.devices('cpu')[0]" == code_cpu
  code_idx: str = adapter.get_device_syntax("'cuda'", device_index="1")
  assert "jax.devices('gpu')[1]" == code_idx
  code_var: str = adapter.get_device_syntax("my_device")
  assert "jax.devices(my_device)[0]" == code_var
