"""
Tests for Extended Inspector logic (Attributes & Classes).

Verifies that the Inspector correctly identifies API members as functions,
classes, or attributes (constants) using a mock Griffe structure.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.discovery.inspector import ApiInspector


class MockGriffeObject:
  """
  Simulates a Griffe object structure for testing.

  Attributes:
      name (str): The object name.
      path (str): Fully qualified path.
      kind (str): 'function', 'attribute', etc.
      parameters (List[MagicMock]): List of mocked parameter objects.
  """

  def __init__(self, name, kind, path, doc=None, members=None, params=None):
    self.name = name
    self.path = path
    self.kind = kind  # "function", "attribute", "class"

    # Griffe Properties
    self.is_function = kind == "function"
    self.is_attribute = kind == "attribute"
    self.is_class = kind == "class"
    self.is_module = kind == "module"
    self.is_alias = False

    self.docstring = MagicMock()
    self.docstring.value = doc

    self.members = members or {}

    # Parameters (for functions)
    # Note: We must explicitly set the .name attribute on the mock,
    # otherwise accessing param.name returns a child Mock object.
    self.parameters = []
    if params:
      for p in params:
        m = MagicMock()
        m.name = p
        self.parameters.append(m)


@pytest.fixture
def mock_package():
  """Builds a mock package tree."""
  # Module constants
  c_pi = MockGriffeObject("pi", "attribute", "math.pi", doc="Ratio")
  c_inf = MockGriffeObject("inf", "attribute", "math.inf")

  # Module function
  f_cos = MockGriffeObject("cos", "function", "math.cos", params=["x"])

  # Class
  c_linear = MockGriffeObject("Linear", "class", "torch.nn.Linear")
  # Method inside class
  m_forward = MockGriffeObject("forward", "function", "torch.nn.Linear.forward", params=["input"])
  c_linear.members = {"forward": m_forward}

  # Root Module
  root = MockGriffeObject("pkg", "module", "pkg")
  root.members = {"pi": c_pi, "inf": c_inf, "cos": f_cos, "Linear": c_linear}

  return root


def test_inspect_identifies_attributes(mock_package, monkeypatch):
  """Verify that attributes (constants) are cataloged with type='attribute'."""

  # Mock griffe.load
  monkeypatch.setattr("griffe.load", lambda x: mock_package)

  inspector = ApiInspector()
  catalog = inspector.inspect("pkg")

  # Check Constant
  assert "math.pi" in catalog
  entry = catalog["math.pi"]
  assert entry["type"] == "attribute"
  assert entry["docstring_summary"] == "Ratio"
  assert entry["params"] == []


def test_inspect_identifies_classes(mock_package, monkeypatch):
  """Verify classes are cataloged as 'class' and recursed."""
  monkeypatch.setattr("griffe.load", lambda x: mock_package)
  inspector = ApiInspector()
  catalog = inspector.inspect("pkg")

  # Check Class Itself is registered
  assert "torch.nn.Linear" in catalog
  assert catalog["torch.nn.Linear"]["type"] == "class"

  # Check Method recursion
  assert "torch.nn.Linear.forward" in catalog
  assert catalog["torch.nn.Linear.forward"]["type"] == "function"


def test_inspect_identifies_functions(mock_package, monkeypatch):
  """Verify standard function cataloging still works."""
  monkeypatch.setattr("griffe.load", lambda x: mock_package)
  inspector = ApiInspector()
  catalog = inspector.inspect("pkg")

  assert "math.cos" in catalog
  assert catalog["math.cos"]["type"] == "function"
  assert catalog["math.cos"]["params"] == ["x"]
