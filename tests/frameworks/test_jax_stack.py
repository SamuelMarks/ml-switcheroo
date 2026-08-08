"""Test suite for the Jax Stack module."""

from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


class MockHighLevelAdapter(JAXStackMixin):
  """Mock High Level Adapter class for testing purposes."""

  pass


def test_get_serialization_imports():
  """Verifies the behavior of get_serialization_imports."""
  adapter = MockHighLevelAdapter()
  imports = adapter.get_serialization_imports()
  assert "import orbax.checkpoint" in imports


def test_get_weight_conversion_imports():
  """Verifies the behavior of get_weight_conversion_imports."""
  adapter = MockHighLevelAdapter()
  imports = adapter.get_weight_conversion_imports()
  assert "import jax.numpy as jnp" in imports
  assert "import orbax.checkpoint" in imports
  assert "from flax.traverse_util import unflatten_dict, flatten_dict" in imports


def test_device_syntax_cuda():
  """Verifies the behavior of device syntax cuda."""
  adapter = MockHighLevelAdapter()
  syntax = adapter.get_device_syntax("'cuda'")
  assert "jax.devices('gpu')[0]" == syntax


def test_device_syntax_cpu_index():
  """Verifies the behavior of device syntax cpu index."""
  adapter = MockHighLevelAdapter()
  syntax = adapter.get_device_syntax("'cpu'", "1")
  assert "jax.devices('cpu')[1]" == syntax


def test_serialization_syntax_orbax():
  """Verifies the behavior of serialization syntax orbax."""
  adapter = MockHighLevelAdapter()

  # test direct call branch
  save_code_direct = adapter.get_serialization_syntax("save", "dir", "state")
  assert "checkpointer().save" in save_code_direct.lower()

  load_code_direct = adapter.get_serialization_syntax("load", "dir")
  assert "checkpointer().restore" in load_code_direct.lower()

  # Use the mixin's newly added weight code
  save_code = adapter.get_weight_save_code("state", "dir")
  assert "PyTreeCheckpointer" in save_code
  assert "tuple_params" in save_code


def test_weight_load_code():
  """Test get weight load code."""
  adapter = MockHighLevelAdapter()
  code = adapter.get_weight_load_code("path")
  assert "orbax.checkpoint.PyTreeCheckpointer" in code
  assert "raw_tree = checkpointer.restore(path)" in code
  assert "flatten_dict" in code


def test_tensor_conversion_syntax():
  """Test tensor conversion syntax."""
  adapter = MockHighLevelAdapter()
  code = adapter.get_tensor_to_numpy_expr("x")
  assert code == "np.array(x)"


def test_weight_save_code():
  """Test weight save code."""
  adapter = MockHighLevelAdapter()
  code = adapter.get_weight_save_code("state", "path")
  assert "tuple_params =" in code
  assert "unflatten_dict(tuple_params)" in code
  assert "orbax.checkpoint.PyTreeCheckpointer" in code
  assert "checkpointer.save(path" in code


def test_wiring_injection():
  """Verifies the behavior of wiring injection."""
  adapter = MockHighLevelAdapter()
  snapshot = {}
  adapter._apply_stack_wiring(snapshot)
  mappings = snapshot.get("mappings", {})
  templates = snapshot.get("templates", {})
  assert "import chex" in templates["import"]
  assert templates["to_numpy"] == "{res_var}"
  assert mappings["Abs"]["api"] == "jnp.abs"
  assert mappings["size"]["requires_plugin"] == "method_to_property"
  assert mappings["Adam"]["api"] == "optax.adam"
  assert mappings["step"]["requires_plugin"] == "optimizer_step"


def test_get_to_numpy_code():
  """Gets to NumPy code."""
  adapter = MockHighLevelAdapter()
  assert "hasattr(obj, '__array__')" in adapter.get_to_numpy_code()


def test_get_device_check_syntax():
  """Gets device check syntax."""
  adapter = MockHighLevelAdapter()
  assert "len(jax.devices('gpu')) > 0" == adapter.get_device_check_syntax()


def test_get_rng_split_syntax():
  """Gets rng split syntax."""
  adapter = MockHighLevelAdapter()
  assert "rng, key = jax.random.split(rng)" == adapter.get_rng_split_syntax("rng", "key")


def test_get_doc_url():
  """Gets documentation URL."""
  adapter = MockHighLevelAdapter()
  assert "jax.numpy.abs.html" in adapter.get_doc_url("jax.numpy.abs")


def test_wiring_injection_existing_templates():
  """Verifies the behavior of wiring injection existing templates."""
  adapter = MockHighLevelAdapter()
  snapshot = {"templates": {"existing": "template"}}
  adapter._apply_stack_wiring(snapshot)
  assert "existing" in snapshot["templates"]
  assert "import" not in snapshot["templates"]


def test_serialization_syntax_invalid_op():
  """Verifies the behavior of serialization syntax invalid op."""
  adapter = MockHighLevelAdapter()
  assert "" == adapter.get_serialization_syntax("invalid", "dir")
