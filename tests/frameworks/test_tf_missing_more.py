def test_tf_adapter_coverage():
  import sys
  from ml_switcheroo.frameworks.base import InitMode, StandardCategory

  # mock tensorflow
  def dummy_func():
    pass

  class DummyLayer:
    pass

  class MockTF:
    class keras:
      class layers:
        Layer1 = DummyLayer

      class losses:
        pass

      class optimizers:
        pass

    class random:
      pass

    class math:
      pass

    class nn:
      relu = dummy_func

  sys.modules["tensorflow"] = MockTF()
  sys.modules["tensorflow.keras"] = MockTF.keras
  sys.modules["tensorflow.keras.layers"] = MockTF.keras.layers
  sys.modules["tensorflow.keras.losses"] = MockTF.keras.losses
  sys.modules["tensorflow.keras.optimizers"] = MockTF.keras.optimizers

  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()
  adapter._mode = InitMode.LIVE

  # Run properties
  _ = adapter.unsafe_submodules
  _ = adapter.search_modules
  _ = adapter.import_alias
  _ = adapter.harness_imports
  _ = adapter.declared_magic_args
  _ = adapter.structural_traits
  _ = adapter.plugin_traits
  _ = adapter.definitions
  _ = adapter.rng_seed_methods

  # Run methods
  _ = adapter.get_harness_init_code()
  _ = adapter.get_to_numpy_code()
  _ = adapter.get_device_syntax("cuda")
  _ = adapter.get_device_syntax("cuda", "i")
  _ = adapter.get_device_syntax("cuda", "1")
  _ = adapter.get_device_check_syntax()
  _ = adapter.get_rng_split_syntax("rng", "key")
  _ = adapter.get_serialization_imports()
  _ = adapter.get_serialization_syntax("op", "f")
  _ = adapter.get_weight_conversion_imports()
  _ = adapter.get_weight_load_code("path")
  _ = adapter.get_tensor_to_numpy_expr("tensor")
  _ = adapter.get_weight_save_code("state", "path")
  _ = adapter.get_doc_url("tf.keras")
  _ = adapter.get_doc_url("other")
  _ = adapter.get_tiered_examples()

  # Collect API
  assert isinstance(adapter.collect_api(StandardCategory.LAYER), list)
  assert isinstance(adapter.collect_api(StandardCategory.LOSS), list)
  assert isinstance(adapter.collect_api(StandardCategory.OPTIMIZER), list)
  assert isinstance(adapter.collect_api(StandardCategory.ACTIVATION), list)

  # Force _collect_ghost
  adapter._mode = InitMode.GHOST
  adapter._snapshot_data = {}
  adapter.collect_api(StandardCategory.LAYER)
  adapter._snapshot_data = {
    "categories": {"layer": [{"api": "api", "api_path": "layer", "name": "layer", "kind": "function", "args": []}]}
  }
  adapter.collect_api(StandardCategory.LAYER)

  # Test apply_wiring
  snap = {"mappings": {"a": {"api": "tensorflow.math.add"}, "b": {"api": "other"}}}
  adapter.apply_wiring(snap)
  assert snap["mappings"]["a"]["api"] == "tf.math.add"

  del sys.modules["tensorflow"]
  del sys.modules["tensorflow.keras"]
  del sys.modules["tensorflow.keras.layers"]
  del sys.modules["tensorflow.keras.losses"]
  del sys.modules["tensorflow.keras.optimizers"]


def test_tf_convert_and_init():
  import sys

  sys.modules["tensorflow"] = None

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.frameworks.tensorflow.load_snapshot_for_adapter", return_value={}
  ):
    from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

    adapter = TensorFlowAdapter()
    assert adapter.convert(123) == 123
