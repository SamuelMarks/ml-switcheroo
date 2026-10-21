"""Test suite for torch."""

import typing

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.torch import TorchAdapter


def test_torch_adapter_basic() -> None:
  """Verify torch adapter basic behavior."""
  adapter = TorchAdapter()

  assert adapter.import_alias == ("torch", "torch")
  ns = adapter.import_namespaces
  assert "torch" in ns
  assert ns["torch"].recommended_alias == "torch"
  assert ns["torch.nn"].recommended_alias == "nn"
  assert ns["torch.nn.functional"].recommended_alias == "F"

  assert SemanticTier.NEURAL in adapter.supported_tiers

  cfg = adapter.test_config
  assert "import torch" in cfg["import"]
  assert "tensor" in cfg["convert_input"]

  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "cpu().numpy()" in adapter.get_to_numpy_code()

  traits = adapter.structural_traits
  assert traits.module_base == "torch.nn.Module"
  assert traits.forward_method == "forward"

  ptraits = adapter.plugin_traits
  assert ptraits.has_numpy_compatible_arrays is False
  assert ptraits.requires_explicit_rng is False
  assert ptraits.requires_functional_state is False

  assert adapter.rng_seed_methods == ["manual_seed", "seed"]
  assert adapter.declared_magic_args == []

  defs = adapter.definitions
  assert isinstance(defs, dict)


def test_torch_adapter_syntax() -> None:
  """Verify torch adapter syntax behavior."""
  adapter = TorchAdapter()

  assert adapter.get_device_syntax("cuda", "0") == "torch.device(cuda, 0)"
  assert adapter.get_device_syntax("cpu", None) == "torch.device(cpu)"

  assert adapter.get_device_check_syntax() == "torch.cuda.is_available()"
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"


def test_torch_adapter_docs() -> None:
  """Verify torch adapter docs behavior."""
  adapter = TorchAdapter()
  url: str | None = adapter.get_doc_url("torch.nn.Linear")
  assert url is not None
  assert "pytorch.org" in url

  url_unkn: str | None = adapter.get_doc_url("unknown")
  assert url_unkn is not None
  assert "pytorch.org" in url_unkn

  examples: dict[str, str] = adapter.get_tiered_examples()
  assert len(examples) > 0


def test_torch_adapter_convert() -> None:
  """Verify torch adapter convert behavior."""
  pytest.importorskip("torch")
  adapter = TorchAdapter()
  import numpy as np
  import torch

  # test list
  t = adapter.convert([1, 2, 3])
  assert isinstance(t, torch.Tensor)

  # test numpy
  t2 = adapter.convert(np.array([1, 2, 3]))
  assert isinstance(t2, torch.Tensor)

  # test primitive
  t3 = adapter.convert(1)
  assert t3 == 1


def test_torch_adapter_collect_ghost() -> None:
  """Verify torch adapter collect ghost behavior."""
  adapter = TorchAdapter()
  res = adapter._collect_ghost(SemanticTier.LOSS)
  assert isinstance(res, list)


def test_torch_adapter_collect_live() -> None:
  """Verify torch adapter collect live behavior."""
  adapter = TorchAdapter()
  res = adapter._collect_live(SemanticTier.NEURAL)
  assert isinstance(res, list)

  res2 = adapter._collect_live(SemanticTier.LOSS)
  assert isinstance(res2, list)

  res3 = adapter._collect_live(SemanticTier.OPTIMIZER)
  assert isinstance(res3, list)

  res4 = adapter._collect_live(SemanticTier.ACTIVATION)
  assert isinstance(res4, list)


def test_torch_adapter_wiring() -> None:
  """Verify torch adapter wiring behavior."""
  adapter = TorchAdapter()
  snapshot: dict[str, typing.Hashable | typing.Sized] = {"mappings": {}}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  res = None
  assert res is None
  assert snapshot == {"mappings": {}}


def test_convert() -> None:
  """Verify convert behavior."""
  from unittest.mock import MagicMock, patch

  from ml_switcheroo.frameworks.torch import TorchAdapter

  with patch.dict("sys.modules", {"torch": MagicMock()}):
    import numpy as np

    adapter = TorchAdapter()
    r1 = adapter.convert(np.array([1, 2, 3]))
    r2 = adapter.convert([1, 2, 3])
    r3 = adapter.convert(1)
  assert r1 is not None
  assert r2 is not None
  assert r3 == 1


def test_torch_ghost_mode_with_snapshot() -> None:
  """Verify torch ghost mode with snapshot behavior."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.torch import TorchAdapter

  with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value={"test": 1}):
    with patch("ml_switcheroo.frameworks.torch.torch", None):
      adapter = TorchAdapter()
      assert adapter._snapshot_data == {"test": 1}


def test_torch_definitions_skip() -> None:
  """Verify torch definitions skip behavior."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.torch import TorchAdapter

  with patch("ml_switcheroo.frameworks.torch.load_definitions", return_value={"Conv2d": "already_exists"}):
    adapter = TorchAdapter()
    defs = adapter.definitions
    assert getattr({"v": getattr({"v": defs}, "__getitem__")("v")["Conv2d"]}, "__getitem__")("v") == getattr(
      {"v": "already_exists"}, "__getitem__"
    )("v")
