def test_harness_generate_template():
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  from pathlib import Path

  hg = HarnessGenerator()

  semantics = {"op1": {"std_args": [["arg1", "int"], {"name": "arg2", "type": "float"}, {"name": "arg3"}]}}

  source = Path("source.py")
  target = Path("target.py")
  out = Path("out.py")
  hg.generate(source, target, out, "jax", "torch", semantics)


def test_harness_adapter_shim_exceptions():
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()

  # Force _build_dynamic_init to return defaults
  with __import__("unittest.mock").mock.patch("ml_switcheroo.testing.harness_generator.get_adapter", return_value=None):
    res = hg._build_dynamic_init("fake_fw")
    assert res == ("", "", "pass")

  # Force _build_result_normalization adapter exceptions
  class MockAdapter:
    def get_to_numpy_code(self):
      raise Exception("Fail")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.harness_generator.get_adapter", return_value=MockAdapter()
  ):
    res = hg._build_result_normalization("jax", "torch")
    assert res == ""


def test_harness_extractor_oserror():
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  import inspect

  hg = HarnessGenerator()

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.utils.code_extractor.CodeExtractor.extract_class", side_effect=OSError("fail")
  ):
    try:
      res = hg._bundle_fuzzer_dependencies()
    except OSError:
      pass


def test_harness_extract_module_functions_oserror():
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  import inspect

  hg = HarnessGenerator()

  # We patch getsource specifically when called by _bundle_fuzzer_dependencies
  original_getsource = inspect.getsource

  def mock_getsource(obj):
    if inspect.isfunction(obj):
      raise OSError("fail")
    return original_getsource(obj)

  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=mock_getsource):
    hg._bundle_fuzzer_dependencies()


def test_harness_build_result_normalization_flax():
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  res = hg._build_result_normalization("flax_nnx", "torch")
  assert "jax" in res or "flax_nnx" in res


def test_harness_generate_adapter_shim_oserror():
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  import inspect

  hg = HarnessGenerator()
  original_getsource = inspect.getsource

  def mock_getsource(obj):
    if hasattr(obj, "__name__") and obj.__name__ == "convert":
      raise OSError("fail")
    return original_getsource(obj)

  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=mock_getsource):
    hg._generate_adapter_shim()


def test_harness_generate_adapter_shim_no_convert():
  from ml_switcheroo.testing.harness_generator import HarnessGenerator
  from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY

  # Temporarily inject an adapter with no convert
  class NoConvertAdapter:
    pass

  _ADAPTER_REGISTRY["fake_fw"] = NoConvertAdapter
  try:
    hg = HarnessGenerator()
    hg._generate_adapter_shim()
  finally:
    del _ADAPTER_REGISTRY["fake_fw"]
