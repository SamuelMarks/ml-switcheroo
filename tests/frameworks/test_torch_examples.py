"""Test suite for the Torch Examples module."""

import ast

import pytest

from ml_switcheroo.frameworks.torch import TorchAdapter


@pytest.fixture
def adapter() -> TorchAdapter:
  """Create a configured `TorchAdapter` fixture for torch examples tests.

  Returns:
      Configured `adapter` fixture value (TorchAdapter).
  """
  return TorchAdapter()


def test_tiered_examples_structure(adapter: TorchAdapter) -> None:
  """Verify `adapter.get_tiered_examples` emits `tier1_math` for tiered examples structure.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural_simple" in examples
  assert "tier2_neural_cnn" in examples
  assert "tier3_extras_dataloader" in examples


def test_tier1_math_validity(adapter: TorchAdapter) -> None:
  """Verify `parse` emits `torch.abs` for tier1 math validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "torch.abs" in code
  assert "torch.add" in code
  assert "torch.mean" in code


def test_tier2_neural_simple_validity(adapter: TorchAdapter) -> None:
  """Verify `parse` emits `class Net(nn.Module):` for tier2 neural simple validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier2_neural_simple"]
  ast.parse(code)
  assert "class Net(nn.Module):" in code
  assert "super().__init__()" in code
  assert "nn.functional.relu" in code


def test_tier2_neural_cnn_validity(adapter: TorchAdapter) -> None:
  """Verify `parse` emits `class ConvNet(nn.Module):` for tier2 neural cnn validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier2_neural_cnn"]
  ast.parse(code)
  assert "class ConvNet(nn.Module):" in code
  assert "nn.Conv2d" in code
  assert "torch.flatten" in code


def test_tier3_extras_dataloader_validity(adapter: TorchAdapter) -> None:
  """Verify `parse` emits `DataLoader` for tier3 extras dataloader validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier3_extras_dataloader"]
  ast.parse(code)
  assert "DataLoader" in code
  assert "TensorDataset" in code
  assert "num_workers=4" in code
