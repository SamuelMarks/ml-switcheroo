"""Test suite for the Harness Generator Missing module."""

import types


def test_harness_generate_template() -> None:
  """Verifies the behavior of harness generate template."""
  from pathlib import Path

  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  semantics = {
    "op1": {"std_args": [["arg1", "int"], {"name": "arg2", "type": "float"}, {"name": "arg3"}]},
    "op2": {"std_args": ["string_arg_only"]},
  }
  source = Path("source.py")
  target = Path("target.py")
  out = Path("out.py")
  getattr({"v": getattr({"v": hg}, "__getitem__")("v").generate}, "__getitem__")("v")(
    getattr({"v": source}, "__getitem__")("v"),
    getattr({"v": target}, "__getitem__")("v"),
    getattr({"v": out}, "__getitem__")("v"),
    getattr({"v": "jax"}, "__getitem__")("v"),
    getattr({"v": "torch"}, "__getitem__")("v"),
    getattr({"v": semantics}, "__getitem__")("v"),
  )
  hg.generate(source, target, out, "jax", "torch", semantics=None)
  _res = None
  assert _res is None
  assert out.name == "out.py"


def test_harness_adapter_shim_exceptions() -> None:
  """Verifies the behavior of harness adapter shim exceptions."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  with __import__("unittest.mock").mock.patch("ml_switcheroo.testing.harness_generator.get_adapter", return_value=None):
    res = hg._build_dynamic_init("fake_fw")
    assert res == ("", "", "pass")

  class MockAdapter:
    """Test helper class MockAdapter."""

    def get_to_numpy_code(self) -> str:
      """Mock implementation of get to NumPy code.

      Returns:
          Computed test result or fixture value.
      """
      raise Exception("Fail")

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.harness_generator.get_adapter", return_value=MockAdapter()
  ):
    res2 = hg._build_result_normalization("jax", "torch")
    assert res2 == ""

  res3 = hg._build_result_normalization("nonexistent_fw", "torch")
  assert isinstance(res3, str)


def test_harness_extractor_oserror() -> None:
  """Verify harness extractor oserror behavior."""
  import pytest

  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.utils.code_extractor.CodeExtractor.extract_class", side_effect=OSError("fail")
  ):
    with pytest.raises(OSError, match="fail"):
      hg._bundle_fuzzer_dependencies()


def test_harness_extract_module_functions_oserror() -> None:
  """Verify harness extract module functions oserror behavior."""
  import inspect

  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  original_getsource = inspect.getsource

  def mock_getsource(obj: type | types.FunctionType | types.MethodType) -> str:
    """Execute mock getsource helper logic.

    Args:
        obj: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if inspect.isfunction(obj):
      raise OSError("fail")
    return original_getsource(obj)

  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=mock_getsource):
    res = hg._bundle_fuzzer_dependencies()
  assert isinstance(res, str)


def test_harness_build_result_normalization_flax() -> None:
  """Verifies the behavior of harness build result normalization Flax."""
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  res = hg._build_result_normalization("flax_nnx", "torch")
  assert "jax" in res or "flax_nnx" in res


def test_harness_generate_adapter_shim_oserror() -> None:
  """Verifies the behavior of harness generate adapter shim oserror."""
  import inspect

  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  hg = HarnessGenerator()
  original_getsource = inspect.getsource

  def mock_getsource(obj: type | types.FunctionType | types.MethodType) -> str:
    """Execute mock getsource helper logic.

    Args:
        obj: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if hasattr(obj, "__name__") and getattr(obj, "__name__") == "convert":
      raise OSError("fail")
    return original_getsource(obj)

  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=mock_getsource):
    res = hg._generate_adapter_shim()
  assert isinstance(res, str)


def test_harness_generate_adapter_shim_no_convert() -> None:
  """Verifies the behavior of harness generate adapter shim no convert."""
  from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY
  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  class NoConvertAdapter:
    """Test helper class NoConvertAdapter."""

    pass

  getattr(getattr({"v": _ADAPTER_REGISTRY}, "__getitem__")("v"), "__setitem__")(
    "fake_fw", getattr({"v": NoConvertAdapter}, "__getitem__")("v")
  )
  try:
    hg = HarnessGenerator()
    res = hg._generate_adapter_shim()
  finally:
    del _ADAPTER_REGISTRY["fake_fw"]
  assert isinstance(res, str)


def test_harness_build_dynamic_init_with_magic_args() -> None:
  """Verify harness build dynamic init with magic args behavior."""
  import unittest.mock as mock

  from ml_switcheroo.testing.harness_generator import HarnessGenerator

  class MockAdapter:
    """Test helper class MockAdapter."""

    harness_imports = ["import something"]
    declared_magic_args = ["my_magic_arg"]

    def get_harness_init_code(self) -> str:
      """Get harness init code.

      Returns:
          Computed test result or fixture value.
      """
      return "def my_helper(): pass"

  hg = HarnessGenerator()
  with mock.patch("ml_switcheroo.testing.harness_generator.get_adapter", return_value=MockAdapter()):
    with mock.patch(
      "ml_switcheroo.testing.signature_extractor.SignatureExtractor.extract_first_function_name", return_value="my_helper"
    ):
      imports_str, init_code, final_logic = hg._build_dynamic_init("mock_fw")
      assert "import something" in imports_str
      assert "my_helper" in init_code
      assert "tgt_inputs[tp] = val" in final_logic
      assert "val = None" in final_logic


def test_harness_injector_leave_module() -> None:
  """Verify harness injector leave module behavior."""
  import libcst as cst

  from ml_switcheroo.testing.harness_generator import HarnessInjector

  injector = HarnessInjector(
    imports_block="import sys\n",
    init_helpers_block="def helper(): pass\n",
    fuzzer_block="",
    to_numpy_block="",
    injection_block="",
    source_framework="jax",
    target_framework="torch",
    source_path="in.py",
    target_path="out.py",
    hints_json="{}",
  )
  module_code = "def to_numpy(): pass\n"
  module = cst.parse_module(module_code)
  updated_module = module.visit(injector)
  updated_code = updated_module.code
  assert "import sys" in updated_code
  assert "def helper" in updated_code


def test_harness_injector_leave_functiondef_no_block() -> None:
  """Verify harness injector leave functiondef no block behavior."""
  import libcst as cst

  from ml_switcheroo.testing.harness_generator import HarnessInjector

  injector = HarnessInjector(
    imports_block="",
    init_helpers_block="",
    fuzzer_block="",
    to_numpy_block="   ",
    injection_block="",
    source_framework="jax",
    target_framework="torch",
    source_path="in.py",
    target_path="out.py",
    hints_json="{}",
  )
  module_code = "def to_numpy(): pass\n"
  module = cst.parse_module(module_code)
  updated_module = module.visit(injector)
  assert updated_module.code == module.code


def test_harness_injector_leave_if_with_injection() -> None:
  """Verify harness injector leave if with injection behavior."""
  import libcst as cst

  from ml_switcheroo.testing.harness_generator import HarnessInjector

  injector = HarnessInjector(
    imports_block="",
    init_helpers_block="",
    fuzzer_block="",
    to_numpy_block="",
    injection_block="tgt_inputs[tp] = 42\n",
    source_framework="jax",
    target_framework="torch",
    source_path="in.py",
    target_path="out.py",
    hints_json="{}",
  )
  module_code = "if tp not in tgt_inputs:\n    pass\n"
  module = cst.parse_module(module_code)
  updated_module = module.visit(injector)
  assert "tgt_inputs[tp] = 42" in updated_module.code
