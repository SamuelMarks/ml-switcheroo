"""
Tests for JAX Stack Wiring and Integration.

Verifies that the `JAXStackMixin` correctly configures high-level framework adapters
with the shared Level 0 (Core) and Level 1 (Common Libs) mappings.
"""

import pytest
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


class MockAdapter(JAXStackMixin):
  """
  A dummy adapter class representing a high-level framework (e.g. Flax, PaxML).
  It inherits from JAXStackMixin to gain wiring capabilities.
  """

  def apply_wiring(self, snapshot):
    # Delegate to the mixin
    self._apply_stack_wiring(snapshot)


@pytest.fixture
def adapter():
  return MockAdapter()


def test_core_math_mappings(adapter):
  """
  Verify Level 0: Array API mappings are injected.
  """
  snapshot = {}
  adapter.apply_wiring(snapshot)

  mappings = snapshot["mappings"]

  # Check 'Abs' normalization
  assert "Abs" in mappings
  assert mappings["Abs"]["api"] == "jnp.abs"

  # Check 'size' method -> property plugin
  assert "size" in mappings
  assert mappings["size"]["api"] == "shape"
  assert mappings["size"]["requires_plugin"] == "method_to_property"


def test_optax_autowiring(adapter):
  """
  Verify Level 1: Optax optimizers are automatically configured.
  """
  snapshot = {}
  adapter.apply_wiring(snapshot)

  mappings = snapshot["mappings"]

  # Check Optimizer mapping (Adam -> optax.adam)
  assert "Adam" in mappings
  assert mappings["Adam"]["api"] == "optax.adam"

  # Check Constructor stripping rule
  assert mappings["Adam"]["requires_plugin"] == "optimizer_constructor"

  # Check Step method mapping
  assert "step" in mappings
  assert mappings["step"]["requires_plugin"] == "optimizer_step"


def test_io_serialization_wiring(adapter):
  """
  Verify Level 1: Orbax syntax generation.
  """
  # Test Imports
  imports = adapter.get_serialization_imports()
  assert "import orbax.checkpoint" in imports[0]

  # Test Save Syntax
  save_code = adapter.get_serialization_syntax(op="save", file_arg="'./ckpt'", object_arg="state")
  assert "orbax.checkpoint.PyTreeCheckpointer().save" in save_code
  assert "directory='./ckpt'" in save_code
  assert "item=state" in save_code

  # Test Load Syntax
  load_code = adapter.get_serialization_syntax(op="load", file_arg="'./ckpt'")
  assert "orbax.checkpoint.PyTreeCheckpointer().restore" in load_code
  assert "('./ckpt')" in load_code


def test_control_flow_templates(adapter):
  """
  Verify Level 0: Control Flow templates are injected.
  """
  snapshot = {}
  adapter.apply_wiring(snapshot)

  templates = snapshot.get("templates", {})

  assert "fori_loop" in templates
  assert "jax.lax.fori_loop" in templates["fori_loop"]
  assert "scan" in templates
  assert "jax.lax.scan" in templates["scan"]


def test_device_syntax_generation(adapter):
  """
  Verify Level 0: Hardware abstraction API.
  """
  # 1. CUDA -> GPU logic
  code_cuda = adapter.get_device_syntax("'cuda'")
  assert "jax.devices('gpu')[0]" == code_cuda

  # 2. CPU logic
  code_cpu = adapter.get_device_syntax("'cpu'")
  assert "jax.devices('cpu')[0]" == code_cpu

  # 3. Explicit Index
  code_idx = adapter.get_device_syntax("'cuda'", device_index="1")
  assert "jax.devices('gpu')[1]" == code_idx

  # 4. Variable argument (formatted via {})
  code_var = adapter.get_device_syntax("my_device")
  assert "jax.devices(my_device)[0]" == code_var
