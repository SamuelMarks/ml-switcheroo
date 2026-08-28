"""Docstring."""

import typing
from ml_switcheroo.frameworks.torch import TorchAdapter
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_torch_adapter_basic() -> None:
  """Docstring."""
  adapter = TorchAdapter()

  assert adapter.import_alias == ("torch", "torch")
  ns: typing.Any = adapter.import_namespaces
  assert "torch" in ns
  assert ns["torch"].recommended_alias == "torch"
  assert ns["torch.nn"].recommended_alias == "nn"
  assert ns["torch.nn.functional"].recommended_alias == "F"

  assert SemanticTier.NEURAL in adapter.supported_tiers

  cfg: dict[str, typing.Any] = adapter.test_config
  assert "import torch" in cfg["import"]
  assert "tensor" in cfg["convert_input"]

  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "cpu().numpy()" in adapter.get_to_numpy_code()

  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "torch.nn.Module"
  assert traits.forward_method == "forward"

  ptraits: typing.Any = adapter.plugin_traits
  assert ptraits.has_numpy_compatible_arrays is False
  assert ptraits.requires_explicit_rng is False
  assert ptraits.requires_functional_state is False

  assert adapter.rng_seed_methods == ["manual_seed", "seed"]
  assert adapter.declared_magic_args == []

  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)


def test_torch_adapter_syntax() -> None:
  """Docstring."""
  adapter = TorchAdapter()

  assert adapter.get_device_syntax("cuda", "0") == "torch.device(cuda, 0)"
  assert adapter.get_device_syntax("cpu", None) == "torch.device(cpu)"

  assert adapter.get_device_check_syntax() == "torch.cuda.is_available()"
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"


def test_torch_adapter_docs() -> None:
  """Docstring."""
  adapter = TorchAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("torch.nn.Linear")
  assert url is not None
  assert "pytorch.org" in url

  url_unkn: typing.Optional[str] = adapter.get_doc_url("unknown")
  assert url_unkn is not None
  assert "pytorch.org" in url_unkn

  examples: dict[str, str] = adapter.get_tiered_examples()
  assert len(examples) > 0


def test_torch_adapter_convert() -> None:
  """Docstring."""
  pytest.importorskip("torch")
  adapter = TorchAdapter()
  import torch  # type: ignore
  import numpy as np

  # test list
  t: typing.Any = adapter.convert([1, 2, 3])
  assert isinstance(t, torch.Tensor)

  # test numpy
  t2: typing.Any = adapter.convert(np.array([1, 2, 3]))
  assert isinstance(t2, torch.Tensor)

  # test primitive
  t3: typing.Any = adapter.convert(1)
  assert t3 == 1


def test_torch_adapter_collect_ghost() -> None:
  """Docstring."""
  adapter = TorchAdapter()
  res: list[typing.Any] = adapter._collect_ghost(SemanticTier.LOSS)
  assert isinstance(res, list)


def test_torch_adapter_collect_live() -> None:
  """Docstring."""
  adapter = TorchAdapter()
  res: list[typing.Any] = adapter._collect_live(SemanticTier.NEURAL)
  assert isinstance(res, list)

  res2: list[typing.Any] = adapter._collect_live(SemanticTier.LOSS)
  assert isinstance(res2, list)

  res3: list[typing.Any] = adapter._collect_live(SemanticTier.OPTIMIZER)
  assert isinstance(res3, list)

  res4: list[typing.Any] = adapter._collect_live(SemanticTier.ACTIVATION)
  assert isinstance(res4, list)


def test_torch_adapter_wiring() -> None:
  """Docstring."""
  adapter = TorchAdapter()
  snapshot: dict[str, typing.Any] = {"mappings": {}}
  adapter.apply_wiring(snapshot)
  pass


def test_convert() -> None:
  """Docstring."""
  import sys
  from unittest.mock import MagicMock
  from ml_switcheroo.frameworks.torch import TorchAdapter

  sys.modules["torch"] = MagicMock()  # type: ignore
  import numpy as np

  adapter = TorchAdapter()
  adapter.convert(np.array([1, 2, 3]))
  adapter.convert([1, 2, 3])
  adapter.convert(1)
  del sys.modules["torch"]
