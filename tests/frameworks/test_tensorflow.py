"""Unit tests for `TensorFlowAdapter` (`ml_switcheroo.frameworks.tensorflow`)."""

import importlib
import sys
import typing
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

import ml_switcheroo.frameworks.tensorflow as tf_mod
from ml_switcheroo.frameworks.base import FrameworkSnapshot, ImportConfig, InitMode, StandardMap
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
from ml_switcheroo.frameworks.tensorflow_examples import get_tf_tiered_examples


def test_tensorflow_adapter_metadata_traits_syntax_and_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `TensorFlowAdapter` metadata, traits, device/serialization/weight syntax, wiring, and definitions.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = TensorFlowAdapter()
  assert adapter.display_name == "TensorFlow"
  assert adapter.ui_priority == 30
  assert adapter.inherits_from is None
  assert adapter.import_alias == ("tensorflow", "tf")
  tf_ns = adapter.import_namespaces["tensorflow"]
  assert isinstance(tf_ns, ImportConfig)
  assert tf_ns.recommended_alias == "tf"
  assert "import tensorflow as tf" in adapter.test_config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "obj.numpy()" in adapter.get_to_numpy_code()
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  assert "set_seed" in adapter.rng_seed_methods

  s_traits = adapter.structural_traits
  assert s_traits.module_base == "keras.Layer"
  assert s_traits.forward_method == "call"

  p_traits = adapter.plugin_traits
  assert p_traits.has_numpy_compatible_arrays is True
  assert p_traits.requires_explicit_rng is False

  assert adapter.get_device_syntax("cuda") == "tf.device('GPU:0')"
  assert adapter.get_device_syntax("cpu") == "tf.device('CPU:0')"
  assert adapter.get_device_syntax("cuda", "1") == "tf.device('GPU:1')"
  assert adapter.get_device_syntax("cuda", "var") == "tf.device(f'GPU:{str(var)}')"
  assert "len(tf.config.list_physical_devices('GPU')) > 0" in adapter.get_device_check_syntax()
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"

  assert adapter.get_serialization_imports() == ["import tensorflow as tf"]
  assert adapter.get_serialization_syntax("save", "f", "obj") == "tf.io.write_file(f, obj)"
  assert adapter.get_serialization_syntax("load", "f") == "tf.io.read_file(f)"
  assert adapter.get_serialization_syntax("save", "f", None) == ""
  assert adapter.get_serialization_syntax("unknown", "f") == ""

  assert adapter.get_weight_conversion_imports() == ["import tensorflow as tf", "import numpy as np"]
  assert "tf.train.load_checkpoint" in adapter.get_weight_load_code("path")
  assert adapter.get_tensor_to_numpy_expr("var") == "np.asarray(var)"
  assert "WARNING: Saving raw dictionary" in adapter.get_weight_save_code("s", "p")

  assert "tf/math/abs" in (adapter.get_doc_url("tensorflow.math.abs") or "")
  assert adapter.get_doc_url("unknown") == "https://www.tensorflow.org/api_docs/python/unknown"
  assert {"tier1_math", "tier2_neural", "tier3_extras"} <= set(get_tf_tiered_examples().keys())
  with patch(
    "ml_switcheroo.frameworks.tensorflow_examples.get_tf_tiered_examples", return_value={"tier2_neural": "code"}
  ):
    assert adapter.get_tiered_examples() == {"tier2_neural": "code"}

  # `apply_wiring` empty and populated mappings
  empty_snap: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(empty_snap)
  assert empty_snap == {}

  mappings_dict: dict[str, JsonValue] = {
    "op1": {"api": "tensorflow.math.add"},
    "op2": {"api": "other.add"},
    "op3": None,
    "op4": {"not_api": 1},
  }
  wired_snap: dict[str, dict[str, JsonValue]] = {"mappings": mappings_dict}
  adapter.apply_wiring(wired_snap)
  op1_entry = mappings_dict["op1"]
  assert isinstance(op1_entry, dict)
  assert op1_entry["api"] == "tf.math.add"
  op2_entry = mappings_dict["op2"]
  assert isinstance(op2_entry, dict)
  assert op2_entry["api"] == "other.add"

  # `definitions`: fallback population vs all pre-existing
  monkeypatch.setattr(tf_mod, "load_definitions", lambda _fw: {})
  assert "Linear" in adapter.definitions

  existing = {k: StandardMap(api=f"tf.{k}") for k in ("Linear", "Conv2d", "Conv1d", "Conv3d", "ConvTranspose2d")}
  monkeypatch.setattr(tf_mod, "load_definitions", lambda _fw: dict(existing))
  assert adapter.definitions["Linear"].api == "tf.Linear"


def test_tensorflow_adapter_modes_collection_and_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify LIVE/GHOST initialization, `_collect_ghost`/`_collect_live`, module reload, and `convert` fallbacks.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(tf_mod, "tf", True)
  assert TensorFlowAdapter()._mode == InitMode.LIVE

  with (
    patch.object(tf_mod, "tf", None),
    patch.object(tf_mod, "load_snapshot_for_adapter", return_value={}),
  ):
    ghost_empty = TensorFlowAdapter()
    assert ghost_empty._mode == InitMode.GHOST
    assert ghost_empty._snapshot_data == FrameworkSnapshot()

  raw_snap: dict[str, JsonValue] = {
    "version": "2.16",
    "categories": {"array_api": [{"name": "add", "api_path": "tf.math.add", "kind": "function"}]},
  }
  with (
    patch.object(tf_mod, "tf", None),
    patch.object(tf_mod, "load_snapshot_for_adapter", return_value=raw_snap),
  ):
    ghost_snap = TensorFlowAdapter()
    assert ghost_snap._snapshot_data == FrameworkSnapshot.from_raw(raw_snap)

  live_adapter = TensorFlowAdapter()

  # `convert` across list, dict, Torch/TF/`__array__` stubs, exception fallback, and missing `tensorflow`
  class _DummyTensor:
    """Stub `tf.Tensor` instance."""

  mock_tf_conv = MagicMock()
  mock_tf_conv.Tensor = _DummyTensor

  def _fake_convert(val: JsonValue) -> _DummyTensor:
    """Convert lists/arrays to `_DummyTensor` and reject other types.

    Args:
        val: Input value.

    Returns:
        `_DummyTensor` instance.

    Raises:
        ValueError: When `val` is neither a list nor an ndarray.
    """
    if type(val).__name__ not in ("list", "ndarray", "str"):
      raise ValueError("unsupported")
    return _DummyTensor()

  mock_tf_conv.convert_to_tensor.side_effect = _fake_convert

  class _DetachStub:
    """Stub tensor with `.detach().cpu().numpy()`."""

    def detach(self) -> "_DetachStub":
      """Return self.

      Returns:
          `_DetachStub` instance.
      """
      return self

    def cpu(self) -> "_DetachStub":
      """Return self.

      Returns:
          `_DetachStub` instance.
      """
      return self

    def numpy(self) -> list[int]:
      """Return `[1, 2]`.

      Returns:
          Integer list `[1, 2]`.
      """
      return [1, 2]

  class _DetachErr:
    """Stub tensor raising `RuntimeError` in `.detach()`."""

    def detach(self) -> typing.Never:
      """Raise `RuntimeError`.

      Returns:
          Never returns; raises `RuntimeError`.

      Raises:
          RuntimeError: Always raised.
      """
      raise RuntimeError("fail")

  class _ArrayErr:
    """Stub object raising `RuntimeError` in `__array__`."""

    def __array__(self) -> typing.Never:
      """Raise `RuntimeError`.

      Returns:
          Never returns; raises `RuntimeError`.

      Raises:
          RuntimeError: Always raised.
      """
      raise RuntimeError("fail")

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "tensorflow", mock_tf_conv)
    assert isinstance(live_adapter.convert([1, 2, 3]), _DummyTensor)
    assert live_adapter.convert({"a": 1}) == {"a": 1}
    ds = _DetachStub()
    assert live_adapter.convert(ds) is ds
    de = _DetachErr()
    assert live_adapter.convert(de) is de
    ae = _ArrayErr()
    assert live_adapter.convert(ae) is ae

    mock_tf_conv.convert_to_tensor.side_effect = RuntimeError("err")
    assert live_adapter.convert([1, 2]) == [1, 2]

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "tensorflow", None)
    assert live_adapter.convert("test") == "test"
    importlib.reload(tf_mod)
    assert tf_mod.tf is None

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "tensorflow", MagicMock())
    importlib.reload(tf_mod)
    assert TensorFlowAdapter()._mode == InitMode.LIVE

  importlib.reload(tf_mod)
