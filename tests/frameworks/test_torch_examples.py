"""
Tests for Torch Adapter Examples.
 Ensures that tiered examples are syntactically valid and contain expected constructs.
"""

import ast
import pytest
from ml_switcheroo.frameworks.torch import TorchAdapter


@pytest.fixture
def adapter():
  return TorchAdapter()


def test_tiered_examples_structure(adapter):
  examples = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural_simple" in examples
  assert "tier2_neural_cnn" in examples
  assert "tier3_extras_dataloader" in examples


def test_tier1_math_validity(adapter):
  code = adapter.get_tiered_examples()["tier1_math"]
  # Check Syntax
  ast.parse(code)
  # Check Logic
  assert "torch.abs" in code
  assert "torch.add" in code
  assert "torch.mean" in code


def test_tier2_neural_simple_validity(adapter):
  code = adapter.get_tiered_examples()["tier2_neural_simple"]
  ast.parse(code)
  # Check it matches the verbatim expectation from integration tests
  assert "class Net(nn.Module):" in code
  assert "super().__init__()" in code
  assert "nn.functional.relu" in code


def test_tier2_neural_cnn_validity(adapter):
  # This is the new expanded example
  code = adapter.get_tiered_examples()["tier2_neural_cnn"]
  ast.parse(code)
  assert "class ConvNet(nn.Module):" in code
  assert "nn.Conv2d" in code
  assert "torch.flatten" in code


def test_tier3_extras_dataloader_validity(adapter):
  code = adapter.get_tiered_examples()["tier3_extras_dataloader"]
  ast.parse(code)
  assert "DataLoader" in code
  assert "TensorDataset" in code
  assert "num_workers=4" in code
