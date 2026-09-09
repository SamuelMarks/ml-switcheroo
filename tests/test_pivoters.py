"""Tests for state pivoters."""

import libcst as cst
from ml_switcheroo.core.rewriter.pivoting.pivoters import (
  OOPToFunctionalPivoter,
  FunctionalToOOPPivoter,
)


def test_oop_to_functional_pivoter():
  """Test lifting OOP state into functional args."""
  source_code = """
class MyModel:
    def forward(self, x):
        return x * self.weight + self.bias + other.weight

    def other(self):
        return self.unmapped
    """
  module = cst.parse_module(source_code)

  mapping = {"weight": "params_weight", "bias": "params_bias"}

  pivoter = OOPToFunctionalPivoter(state_mapping=mapping, inject_rng=True)
  modified = module.visit(pivoter)
  code = modified.code

  assert "self.weight" not in code
  assert "params_weight" in code
  assert "self.bias" not in code
  assert "params_bias" in code
  assert "self.unmapped" in code
  assert "other.weight" in code
  assert "def forward(self, x, rng):" in code
  assert "def other(self):" in code


def test_oop_to_functional_rng_already_present():
  """Test OOPToFunctionalPivoter doesn't inject rng if already present."""
  source_code = """
class MyModel:
    def __call__(self, x, rng):
        return x * self.w
    """
  module = cst.parse_module(source_code)
  pivoter = OOPToFunctionalPivoter(state_mapping={"w": "params_w"}, inject_rng=True)
  modified = module.visit(pivoter)
  code = modified.code

  assert "def __call__(self, x, rng):" in code
  # Ensure it didn't duplicate it
  assert code.count("rng") == 1


def test_functional_to_oop_pivoter():
  """Test mapping functional args back into OOP state."""
  source_code = """
def forward(self, x, rng):
    y = x * params_weight + params_bias
    z = params_unmapped
    return y

def __call__(self):
    pass

def method_only_rng(rng):
    return 1
    """
  module = cst.parse_module(source_code)

  mapping = {"params_weight": "weight", "params_bias": "bias"}

  pivoter = FunctionalToOOPPivoter(state_mapping=mapping, drop_rng=True)
  modified = module.visit(pivoter)
  code = modified.code

  assert "self.weight" in code
  assert "self.bias" in code
  assert "params_weight" not in code
  assert "params_bias" not in code
  assert "params_unmapped" in code
  assert "def forward(self, x):" in code
  assert "def __call__(self):" in code
  assert "def method_only_rng():" in code


def test_functional_to_oop_pivoter_no_drop():
  """Test FunctionalToOOPPivoter when drop_rng is False."""
  source_code = """
def forward(self, x, rng):
    return x
    """
  module = cst.parse_module(source_code)
  pivoter = FunctionalToOOPPivoter(state_mapping={}, drop_rng=False)
  modified = module.visit(pivoter)
  code = modified.code
  assert "def forward(self, x, rng):" in code
