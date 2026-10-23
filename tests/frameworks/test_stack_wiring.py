"""Test suite for the Stack Wiring module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

import pytest

from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


class MockAdapter(JAXStackMixin):
  """Test helper class MockAdapter."""

  def apply_wiring(self, snapshot: dict[str, JsonValue]) -> None:
    """Mock implementation of apply wiring.

    Args:
        snapshot: The snapshot (dict[str, JsonValue]) for apply wiring.
    """
    getattr({"v": getattr({"v": self}, "__getitem__")("v")._apply_stack_wiring}, "__getitem__")("v")(
      getattr({"v": snapshot}, "__getitem__")("v")
    )


@pytest.fixture
def adapter() -> MockAdapter:
  """Execute adapter helper logic.

  Returns:
      Configured `adapter` fixture value (MockAdapter).
  """
  return MockAdapter()


def test_core_math_mappings(adapter: MockAdapter) -> None:
  """Verifies the behavior of core math mappings.

  Args:
      adapter: The adapter (MockAdapter) for test core math mappings.
  """
  snapshot: dict[str, RuntimeTensorOrValue] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  mappings = snapshot["mappings"]
  assert getattr({"v": "Adam"}, "__getitem__")("v") in getattr({"v": mappings}, "__getitem__")("v")
  assert getattr(
    {"v": getattr({"v": getattr({"v": mappings}, "__getitem__")("v")["Adam"]}, "__getitem__")("v")["api"]}, "__getitem__"
  )("v") == getattr({"v": "optax.adam"}, "__getitem__")("v")
  assert getattr({"v": "step"}, "__getitem__")("v") in getattr({"v": mappings}, "__getitem__")("v")
  assert getattr(
    {"v": getattr({"v": getattr({"v": mappings}, "__getitem__")("v")["step"]}, "__getitem__")("v")["requires_plugin"]},
    "__getitem__",
  )("v") == getattr({"v": "optimizer_step"}, "__getitem__")("v")


def test_optax_autowiring(adapter: MockAdapter) -> None:
  """Verifies the behavior of optax autowiring.

  Args:
      adapter: The adapter (MockAdapter) for test optax autowiring.
  """
  snapshot: dict[str, RuntimeTensorOrValue] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  mappings = snapshot["mappings"]
  assert getattr({"v": "Adam"}, "__getitem__")("v") in getattr({"v": mappings}, "__getitem__")("v")
  assert getattr(
    {"v": getattr({"v": getattr({"v": mappings}, "__getitem__")("v")["Adam"]}, "__getitem__")("v")["api"]}, "__getitem__"
  )("v") == getattr({"v": "optax.adam"}, "__getitem__")("v")
  assert getattr(
    {"v": getattr({"v": getattr({"v": mappings}, "__getitem__")("v")["Adam"]}, "__getitem__")("v")["requires_plugin"]},
    "__getitem__",
  )("v") == getattr({"v": "optimizer_constructor"}, "__getitem__")("v")
  assert getattr({"v": "step"}, "__getitem__")("v") in getattr({"v": mappings}, "__getitem__")("v")
  assert getattr(
    {"v": getattr({"v": getattr({"v": mappings}, "__getitem__")("v")["step"]}, "__getitem__")("v")["requires_plugin"]},
    "__getitem__",
  )("v") == getattr({"v": "optimizer_step"}, "__getitem__")("v")


def test_io_serialization_wiring(adapter: MockAdapter) -> None:
  """Verifies the behavior of I/O serialization wiring.

  Args:
      adapter: The adapter (MockAdapter) for test io serialization wiring.
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
      adapter: The adapter (MockAdapter) for test control flow templates.
  """
  snapshot: dict[str, RuntimeTensorOrValue] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  templates = snapshot.get("templates", {})
  assert getattr({"v": "fori_loop"}, "__getitem__")("v") in getattr({"v": templates}, "__getitem__")("v")
  assert getattr({"v": "jax.lax.fori_loop"}, "__getitem__")("v") in getattr(
    {"v": getattr({"v": templates}, "__getitem__")("v")["fori_loop"]}, "__getitem__"
  )("v")
  assert getattr({"v": "scan"}, "__getitem__")("v") in getattr({"v": templates}, "__getitem__")("v")
  assert getattr({"v": "jax.lax.scan"}, "__getitem__")("v") in getattr(
    {"v": getattr({"v": templates}, "__getitem__")("v")["scan"]}, "__getitem__"
  )("v")


def test_device_syntax_generation(adapter: MockAdapter) -> None:
  """Verifies the behavior of device syntax generation.

  Args:
      adapter: The adapter (MockAdapter) for test device syntax generation.
  """
  code_cuda: str = adapter.get_device_syntax("'cuda'")
  assert "jax.devices('gpu')[0]" == code_cuda
  code_cpu: str = adapter.get_device_syntax("'cpu'")
  assert "jax.devices('cpu')[0]" == code_cpu
  code_idx: str = adapter.get_device_syntax("'cuda'", device_index="1")
  assert "jax.devices('gpu')[1]" == code_idx
  code_var: str = adapter.get_device_syntax("my_device")
  assert "jax.devices(my_device)[0]" == code_var
