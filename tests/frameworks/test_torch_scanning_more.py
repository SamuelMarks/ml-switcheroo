class DummyModule:
  pass


class DummyOptimizer:
  pass


class MyLoss(DummyModule):
  pass


class MyOpt(DummyOptimizer):
  pass


class ReLU(DummyModule):
  pass


class MockTorchNNModulesActivation:
  ReLU = ReLU


class MockTorchNNFunctional:
  @staticmethod
  def relu():
    pass


class MockTorchNN:
  Module = DummyModule
  MyLoss = MyLoss
  ReLU = ReLU
  modules = type("modules", (), {"activation": MockTorchNNModulesActivation})
  functional = MockTorchNNFunctional


class MockTorchOptim:
  Optimizer = DummyOptimizer
  MyOpt = MyOpt


class MockTorch:
  nn = MockTorchNN
  optim = MockTorchOptim


def test_torch_scanning_coverage():
  import sys
  from ml_switcheroo.frameworks.torch import TorchAdapter
  from ml_switcheroo.frameworks.base import InitMode

  adapter = TorchAdapter()
  adapter._mode = InitMode.LIVE

  sys.modules["torch"] = MockTorch()
  sys.modules["torch.nn"] = MockTorch.nn
  sys.modules["torch.nn.modules.activation"] = MockTorchNNModulesActivation
  sys.modules["torch.nn.functional"] = MockTorchNNFunctional
  sys.modules["torch.optim"] = MockTorch.optim

  import ml_switcheroo.frameworks.torch as t

  t.torch = MockTorch()
  t.nn = MockTorch.nn
  t.optim = MockTorch.optim

  adapter._scan_losses()
  adapter._scan_optimizers()
  adapter._scan_activations()
  adapter._scan_layers()

  t.nn = None
  t.optim = None
  adapter._scan_losses()
  adapter._scan_optimizers()
  adapter._scan_layers()

  # test doc url
  adapter.get_doc_url("torch.nn.init")

  # import error in activations
  t.nn = MockTorch.nn
  sys.modules["torch.nn.modules.activation"] = None
  sys.modules["torch.nn.functional"] = None
  try:
    adapter._scan_activations()
  finally:
    del sys.modules["torch.nn.modules.activation"]
    del sys.modules["torch.nn.functional"]

  with __import__("unittest.mock").mock.patch("ml_switcheroo.frameworks.torch.load_definitions", return_value={}):
    adapter.definitions
