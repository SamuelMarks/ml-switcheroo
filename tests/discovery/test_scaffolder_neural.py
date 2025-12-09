"""
Tests for Spec-Driven Neural Scaffolding.

Verifies that:
1. APIs matching loaded NEURAL Specs are categorized as Tier B (Neural).
2. Structural heuristics fallback still works using DYNAMIC adapter config.
"""

import json
from unittest.mock import patch, MagicMock

from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier

# --- Mocks ---


class MockInspector:
  def inspect(self, fw):
    if fw == "torch":
      return {
        # Case 1: Matches ONNX 'Relu' exactly (Case insensitive)
        "torch.nn.ReLU": {"name": "ReLU", "type": "class", "params": []},
        # Case 2: Matches ONNX 'Gemm' (Hypothetically mapped to Linear)
        "torch.nn.Linear": {"name": "Linear", "type": "class", "params": []},
        # Case 3: Does NOT match Spec, matches Regex (e.g. .custom_n. is Neural)
        "torch.custom_n.UnknownLayer": {"name": "UnknownLayer", "type": "class", "params": []},
        # Case 4: Matches Math Spec
        "torch.abs": {"name": "abs", "type": "function", "params": ["x"]},
      }
    return {}


class MockSemantics(SemanticsManager):
  def __init__(self):
    # We manually populate data and origins to simulate loaded Specs
    self.data = {
      "Relu": {"std_args": ["x"], "variants": {}},
      "Conv2d": {"std_args": ["x", "w"], "variants": {}},
      "abs": {"std_args": ["x"], "variants": {}},
    }

    # Simulate Origins
    self._key_origins = {
      "Relu": SemanticTier.NEURAL.value,
      "Conv2d": SemanticTier.NEURAL.value,
      "abs": SemanticTier.ARRAY_API.value,
    }


# --- Tests ---


def test_spec_driven_categorization(tmp_path):
  """
  Verify that discovered APIs are sorted into JSONs based on the Semantics Spec.
  """
  semantics = MockSemantics()
  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.inspector = MockInspector()

  scaffolder.scaffold(["torch"], tmp_path)

  # Load Outputs
  neural_file = tmp_path / "k_neural_net.json"
  math_file = tmp_path / "k_array_api.json"

  with open(neural_file, "rt", encoding="utf-8") as f:
    neural_data = json.load(f)

  with open(math_file, "rt", encoding="utf-8") as f:
    math_data = json.load(f)

  # 1. Spec Match Matches
  assert "Relu" in neural_data
  assert neural_data["Relu"]["variants"]["torch"]["api"] == "torch.nn.ReLU"

  # 3. Spec Match: abs (Array) -> matches torch.abs
  assert "abs" in math_data
  assert "abs" not in neural_data


def test_heuristic_fallback_dynamic(tmp_path):
  """
  Verify that if no spec matches, we use Dynamic Regex from adapter.
  """
  semantics = MockSemantics()
  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.inspector = MockInspector()

  # Mock available frameworks to include 'torch'
  with patch("ml_switcheroo.discovery.scaffolder.available_frameworks", return_value=["torch"]):
    # Mock the adapter to return custom regex patterns
    mock_adapter = MagicMock()
    # We define a custom pattern ".custom_n." to prove we aren't using hardcoded .nn values
    mock_adapter.discovery_heuristics = {"neural": [r"\.custom_n\.", r"\.nn\."], "extras": []}

    with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=mock_adapter):
      scaffolder.scaffold(["torch"], tmp_path)

  neural_file = tmp_path / "k_neural_net.json"
  with open(neural_file, "rt", encoding="utf-8") as f:
    neural_data = json.load(f)

  # 'UnknownLayer' lives in 'torch.custom_n.UnknownLayer'.
  # Regex `\.custom_n\.` should catch it.
  assert "UnknownLayer" in neural_data
  assert neural_data["UnknownLayer"]["variants"]["torch"]["api"] == "torch.custom_n.UnknownLayer"
