"""Test suite for the Stack Wiring module."""

import pytest
from pydantic import JsonValue

from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


class MockAdapter(JAXStackMixin):
  """Stub `JAXStackMixin` subclass implementing `apply_wiring` for stack wiring tests."""

  def apply_wiring(self, snapshot: dict[str, dict[str, JsonValue]]) -> None:
    """Delegate wiring application to `_apply_stack_wiring`.

    Args:
        snapshot: Mapping snapshot dictionary to populate (`dict[str, dict[str, JsonValue]]`).
    """
    self._apply_stack_wiring(snapshot)


@pytest.fixture
def adapter() -> MockAdapter:
  """Create a configured `MockAdapter` fixture for stack wiring tests.

  Returns:
      Configured `adapter` fixture value (MockAdapter).
  """
  return MockAdapter()


def test_core_math_mappings(adapter: MockAdapter) -> None:
  """Verify core math mappings produces the expected state.

  Args:
      adapter: Input `adapter` parameter value (`MockAdapter`).
  """
  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snapshot)
  mappings = snapshot["mappings"]
  assert "Adam" in mappings
  adam = mappings["Adam"]
  assert isinstance(adam, dict)
  assert adam["api"] == "optax.adam"
  assert "step" in mappings
  step = mappings["step"]
  assert isinstance(step, dict)
  assert step["requires_plugin"] == "optimizer_step"


def test_optax_autowiring(adapter: MockAdapter) -> None:
  """Verify optax autowiring produces the expected state.

  Args:
      adapter: Input `adapter` parameter value (`MockAdapter`).
  """
  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snapshot)
  mappings = snapshot["mappings"]
  assert "Adam" in mappings
  adam = mappings["Adam"]
  assert isinstance(adam, dict)
  assert adam["api"] == "optax.adam"
  assert adam["requires_plugin"] == "optimizer_constructor"
  assert "step" in mappings
  step = mappings["step"]
  assert isinstance(step, dict)
  assert step["requires_plugin"] == "optimizer_step"


def test_io_serialization_wiring(adapter: MockAdapter) -> None:
  """Verify I/O serialization wiring works as expected.

  Args:
      adapter: Input `adapter` parameter value (`MockAdapter`).
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
  """Verify `snapshot.get` handles control flow templates as expected.

  Args:
      adapter: Input `adapter` parameter value (`MockAdapter`).
  """
  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snapshot)
  templates = snapshot.get("templates", {})
  assert "fori_loop" in templates
  fori_loop = templates["fori_loop"]
  assert isinstance(fori_loop, str)
  assert "jax.lax.fori_loop" in fori_loop
  assert "scan" in templates
  scan = templates["scan"]
  assert isinstance(scan, str)
  assert "jax.lax.scan" in scan


def test_device_syntax_generation(adapter: MockAdapter) -> None:
  """Verify `adapter.get_device_syntax` handles device syntax generation as expected.

  Args:
      adapter: Input `adapter` parameter value (`MockAdapter`).
  """
  code_cuda: str = adapter.get_device_syntax("'cuda'")
  assert code_cuda == "jax.devices('gpu')[0]"
  code_cpu: str = adapter.get_device_syntax("'cpu'")
  assert code_cpu == "jax.devices('cpu')[0]"
  code_idx: str = adapter.get_device_syntax("'cuda'", device_index="1")
  assert code_idx == "jax.devices('gpu')[1]"
  code_var: str = adapter.get_device_syntax("my_device")
  assert code_var == "jax.devices(my_device)[0]"
