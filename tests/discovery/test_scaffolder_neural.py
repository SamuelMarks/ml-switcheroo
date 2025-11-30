"""
Tests for Spec-Driven Neural Scaffolding.

Verifies that:
1. APIs matching loaded NEURAL Specs are categorized as Tier B (Neural),
   even if they don't reside in `.nn`.
2. APIs matching loaded MATH Specs are categorized as Tier A (Math).
3. Structural heuristics fallback still works for unknown APIs.
"""

import json
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
        # If spec says 'Linear', we match 'Linear'.
        "torch.nn.Linear": {"name": "Linear", "type": "class", "params": []},
        # Case 3: Does NOT match Spec, but has .nn (Heuristic)
        "torch.nn.UnknownLayer": {"name": "UnknownLayer", "type": "class", "params": []},
        # Case 4: Matches Math Spec
        "torch.abs": {"name": "abs", "type": "function", "params": ["x"]},
        # Case 5: Weird path but matches Spec Name
        # e.g. A layer defined in a utils folder but named 'Conv2d'
        "torch.utils.custom.Conv2d": {"name": "Conv2d", "type": "class", "params": []},
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

  # 1. Spec Match: Relu (ONNX) -> matches torch.nn.ReLU
  # Note: Scaffolder uses Abstract ID 'Relu' from spec
  assert "Relu" in neural_data
  assert neural_data["Relu"]["variants"]["torch"]["api"] == "torch.nn.ReLU"

  # 2. Spec Match: Conv2d (ONNX) -> matches torch.utils.custom.Conv2d
  # Even though path is weird, the Name matches the Spec, so it goes to Neural.
  assert "Conv2d" in neural_data
  assert neural_data["Conv2d"]["variants"]["torch"]["api"] == "torch.utils.custom.Conv2d"

  # 3. Spec Match: abs (Array) -> matches torch.abs
  assert "abs" in math_data
  assert "abs" not in neural_data


def test_heuristic_fallback(tmp_path):
  """
  Verify that if no spec matches, we fall back to checking '.nn' path.
  """
  semantics = MockSemantics()
  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.inspector = MockInspector()

  scaffolder.scaffold(["torch"], tmp_path)

  neural_file = tmp_path / "k_neural_net.json"
  with open(neural_file, "rt", encoding="utf-8") as f:
    neural_data = json.load(f)

  # 'UnknownLayer' is not in MockSemantics._key_origins.
  # But it lives in 'torch.nn.UnknownLayer'.
  # Heuristic should catch it.
  assert "UnknownLayer" in neural_data
  assert neural_data["UnknownLayer"]["variants"]["torch"]["api"] == "torch.nn.UnknownLayer"


def test_spec_overrides_heuristic(tmp_path):
  """
  Scenario: An API matches a Math Spec but lives in a Neural-looking path.
  Expectation: Spec wins (goes to Array API).
  """

  # Create a conflict scenario
  class TrickyInspector:
    def inspect(self, _fw):
      return {"torch.nn.functional.abs": {"name": "abs", "type": "function", "params": ["x"]}}

  # 'abs' is defined as Math in MockSemantics
  semantics = MockSemantics()

  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.inspector = TrickyInspector()

  scaffolder.scaffold(["torch"], tmp_path)

  math_file = tmp_path / "k_array_api.json"
  neural_file = tmp_path / "k_neural_net.json"

  with open(math_file, "rt", encoding="utf-8") as f:
    math_data = json.load(f)

  # It should be in Math because 'abs' matches a Math Spec ID.
  assert "abs" in math_data
  assert math_data["abs"]["variants"]["torch"]["api"] == "torch.nn.functional.abs"

  # Verify it didn't duplicate into Neural just because of path
  if neural_file.exists():
    with open(neural_file, "rt", encoding="utf-8") as f:
      neural_data = json.load(f)
    assert "abs" not in neural_data
