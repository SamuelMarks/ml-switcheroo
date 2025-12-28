"""
Tests for the Framework Auto-Injector.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.tools.injector_fw import FrameworkInjector


@pytest.fixture
def sample_variant():
  """Returns a fully populated variant configuration."""
  return FrameworkVariant(
    api="torch.nn.functional.log_softmax",
    args={"dim": "dim"},
    requires_plugin="custom_plugin",
  )


@pytest.fixture
def target_source():
  """
  Returns python source code representing a framework adapter file.
  Includes a matching adapter ('torch') and a non-matching one ('jax').
  """
  return """
from typing import Dict
from ml_switcheroo.frameworks.base import register_framework, StandardMap

@register_framework("jax")
class JaxAdapter:
    @property
    def definitions(self):
        return {"Existing": StandardMap(api="foo")}

@register_framework("torch")
class TorchAdapter:
    # Some other methods
    def convert(self, x):
        return x

    @property
    def definitions(self) -> Dict[str, StandardMap]:
        return {
            "Abs": StandardMap(api="torch.abs"),
            "Relu": StandardMap(api="torch.relu")
        }
"""


def test_injector_targets_correct_class(target_source, sample_variant):
  """
  Verify the injector modifies 'TorchAdapter' based on the 'torch' key.
  """
  wrapper = cst.parse_module(target_source)
  transformer = FrameworkInjector("torch", "LogSoftmax", sample_variant)
  new_module = wrapper.visit(transformer)

  assert transformer.found is True

  code = new_module.code

  # Check that LogSoftmax is inserted
  assert '"LogSoftmax": StandardMap(api="torch.nn.functional.log_softmax"' in code

  # Check that keywords are used correctly
  assert 'args={"dim": "dim"}' in code
  assert 'requires_plugin="custom_plugin"' in code

  # Ensure we didn't touch JAX
  assert code.count('"LogSoftmax"') == 1


def test_import_injection_heuristic_disabled(target_source):
  """
  Scenario: Variant has API 'scipy.special.erf'.
  Expectation: Injector should NOT inject 'import scipy' based on heuristic.
  This validates the fix for the 'import tf' bug.
  """
  scipy_var = FrameworkVariant(api="scipy.special.erf")
  wrapper = cst.parse_module(target_source)
  transformer = FrameworkInjector("torch", "Erf", scipy_var)
  new_module = wrapper.visit(transformer)

  code = new_module.code

  # 1. Check mapping was injected
  assert '"Erf": StandardMap(api="scipy.special.erf")' in code

  # 2. Check import was NOT injected
  assert "import scipy" not in code


def test_explicit_import_injection(target_source):
  """
  Scenario: Variant has 'required_imports=["import scipy"]'.
  Expectation: 'import scipy' IS injected at top of file.
  """
  scipy_var = FrameworkVariant(api="scipy.special.erf", required_imports=["import scipy"])
  wrapper = cst.parse_module(target_source)
  transformer = FrameworkInjector("torch", "Erf", scipy_var)
  new_module = wrapper.visit(transformer)

  code = new_module.code

  # Check import injection
  assert "import scipy" in code
  # Ensure it is at the top (before class def)
  assert code.find("import scipy") < code.find("@register_framework")
