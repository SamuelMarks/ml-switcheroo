"""
Tests for LaTeX Emitter.

Verifies:
1.  Python Code -> LaTeX string flow.
2.  Identification of Layers (__init__) vs Ops (forward).
3.  Handling of positional and keyword arguments.
4.  Valid document structure.
5.  Instructional comment presence.
"""

import pytest
from ml_switcheroo.core.latex.emitter import LatexEmitter


@pytest.fixture
def emitter():
  return LatexEmitter()


def test_basic_model_emission(emitter):
  """
  Scenario: Simple Linear model.
  """
  code = """ 
import torch.nn as nn
class Net(nn.Module): 
    def __init__(self): 
        self.fc = nn.Linear(10, 5) 

    def forward(self, x): 
        return self.fc(x) 
"""
  latex = emitter.emit(code, model_name="Simple")

  # 1. Structure
  assert r"\documentclass" in latex
  assert r"\begin{DefModel}{Simple}" in latex

  # 2. Attribute (Memory)
  # GraphExtractor extracts args as arg_0, arg_1
  # Expect: \Attribute{fc}{Linear}{arg_0=10, arg_1=5}
  assert r"\Attribute{fc}{Linear}" in latex

  # 3. Input
  # GraphExtractor normalizes the input provenance ID to "input" (standard graph convention)
  # regardless of the python variable name "x".
  assert r"\Input{input}{[_]}" in latex

  # 4. StateOp
  # The operation arg points to the source node ID "input"
  assert r"\StateOp{op_fc}{fc}{input}{[_]}" in latex

  # 5. Return
  assert r"\Return{op_fc}" in latex

  # 6. Usage Comment (New Requirement)
  assert "% [Requirement] midl.sty" in latex
  assert "% Ensure 'midl.sty' is in your LaTeX path." in latex


def test_mixed_functional_emission(emitter):
  """
  Scenario: Layer call then Functional Call (ReLU).
  """
  code = """ 
class Mix(nn.Module): 
    def __init__(self): 
        self.conv = nn.Conv2d(1, 1) 

    def forward(self, img): 
        x = self.conv(img) 
        x = F.relu(x) 
        return x
"""
  latex = emitter.emit(code)

  # Check Attribute
  assert r"\Attribute{conv}{Conv2d}" in latex

  # Check StateOp
  # 'img' variable maps to 'input' node ID in GraphExtractor
  assert r"\StateOp{op_conv}{conv}{input}{[_]}" in latex

  # Check ComputeOp (functional relu)
  # The ID generator uses op_NODEID.
  # GraphExtractor heuristics create functional node IDs like 'func_relu'
  # Emitter normalizes type to 'Relu'
  assert r"\Op{op_func_relu}{Relu}" in latex

  # Check Flow: Relu should take op_conv as arg
  assert "{op_conv}" in latex


def test_keyword_arguments_emission(emitter):
  """
  Verify keyword args in Python propagate to LaTeX config.
  """
  code = """ 
class KArg(nn.Module): 
    def __init__(self): 
        self.pool = nn.MaxPool2d(kernel_size=2) 
    def forward(self, x): 
        return self.pool(x) 
"""
  latex = emitter.emit(code)

  assert r"\Attribute{pool}{MaxPool2d}" in latex
  assert "kernel_size=2" in latex
