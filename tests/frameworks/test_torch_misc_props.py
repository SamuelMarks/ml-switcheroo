def test_torch_misc_props():
  from ml_switcheroo.frameworks.torch import TorchAdapter

  adapter = TorchAdapter()
  _ = adapter.discovery_heuristics
  _ = adapter.supported_tiers
  _ = adapter.test_config

  with __import__("unittest.mock").mock.patch("ml_switcheroo.frameworks.torch.issubclass", side_effect=TypeError):
    adapter._scan_optimizers()


def test_torch_serialization():
  from ml_switcheroo.frameworks.torch import TorchAdapter

  adapter = TorchAdapter()
  _ = adapter.get_serialization_syntax("save", "f", "obj")
  _ = adapter.get_serialization_syntax("load", "f")


def test_torch_import_namespaces():
  from ml_switcheroo.frameworks.torch import TorchAdapter

  adapter = TorchAdapter()
  _ = adapter.import_namespaces
