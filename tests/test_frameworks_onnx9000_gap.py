from unittest import mock


def test_onnx9000_gap():
  import importlib
  import ml_switcheroo.frameworks.onnx9000 as fonnx

  # Simulate missing ir
  with mock.patch.dict("sys.modules", {"onnx9000.core": None}):
    importlib.reload(fonnx)
    adapter = fonnx.Onnx9000Framework()
    assert adapter.search_modules == []
    assert adapter.unsafe_submodules == set()

    traits = adapter.plugin_traits
    assert not traits.has_numpy_compatible_arrays
    assert traits.strict_materialization_method is None

  # Simulate ir present
  with mock.patch.dict("sys.modules", {"onnx9000.core.ir": mock.MagicMock()}):
    importlib.reload(fonnx)
    adapter2 = fonnx.Onnx9000Framework()
    assert adapter2.search_modules == ["onnx9000.core.ir"]


def test_onnx9000_methods():
  import ml_switcheroo.frameworks.onnx9000 as fonnx
  import importlib

  importlib.reload(fonnx)
  adapter = fonnx.Onnx9000Framework()
  assert adapter.collect_api("foo") == []
  adapter.apply_wiring({})
  assert adapter.convert("data") == "data"
  assert adapter.get_device_syntax("cuda") == "None"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == ["import onnx9000.core.exporter"]
  assert "onnx9000.core.exporter.export(obj, file)" == adapter.get_serialization_syntax("save", "file", "obj")
  assert "onnx9000.core.parser.parse(file)" == adapter.get_serialization_syntax("load", "file")
  assert adapter.get_serialization_syntax("other", "file") == ""
  assert adapter.get_weight_conversion_imports() == ["from onnx9000.core import parser"]
  assert "model = parser.parse(my_path)" in adapter.get_weight_load_code("my_path")
  assert adapter.get_tensor_to_numpy_expr("t") == "t.to_numpy()"
  assert "exporter.export_weights(st, p)" in adapter.get_weight_save_code("st", "p")
  assert adapter.get_doc_url("Node") == "https://github.com/SamuelMarks/onnx9000"
