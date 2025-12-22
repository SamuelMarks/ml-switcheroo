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
  Verify the injector modifies 'TorchAdapter' based on the 'torch' key,
  ignoring 'JaxAdapter'. Checks for strict keyword formatting (key=val).
  """
  wrapper = cst.parse_module(target_source)
  transformer = FrameworkInjector("torch", "LogSoftmax", sample_variant)
  new_module = wrapper.visit(transformer)

  assert transformer.found is True

  code = new_module.code

  # Check that LogSoftmax is inserted
  assert '"LogSoftmax": StandardMap(api="torch.nn.functional.log_softmax"' in code

  # Check that keywords are used correctly (not quoted string keys, but py args)
  # Expected: args={"dim": "dim"}
  assert 'args={"dim": "dim"}' in code

  # Expected: requires_plugin="custom_plugin"
  assert 'requires_plugin="custom_plugin"' in code

  # Ensure we didn't touch JAX (should only appear once in Torch section)
  # Since JaxAdapter definitions is empty except "Existing",
  # we verify LogSoftmax count matches our single injection.
  assert code.count('"LogSoftmax"') == 1


def test_injector_with_casts(target_source):
  """
  Verify that casts configuration is injected into StandardMap arguments.
  """
  cast_variant = FrameworkVariant(api="some.api", casts={"arg1": "int32", "arg2": "float32"})
  wrapper = cst.parse_module(target_source)
  transformer = FrameworkInjector("torch", "CastOp", cast_variant)
  new_module = wrapper.visit(transformer)

  code = new_module.code
  assert '"CastOp": StandardMap(api="some.api"' in code

  # Note: dict key order isn't guaranteed by python strictly for sets,
  # but here we construct CST dict. Logic iterates dict.
  # We check existence of the dict structure.
  assert 'casts={"arg1": "int32", "arg2": "float32"}' in code or 'casts={"arg2": "float32", "arg1": "int32"}' in code


def test_injector_with_inject_args(target_source):
  """
  Verify that inject_args configuration is injected into StandardMap arguments.
  """
  # Variant with injected arguments
  inject_variant = FrameworkVariant(api="target.op", inject_args={"epsilon": 1e-5, "flag": True})

  wrapper = cst.parse_module(target_source)
  transformer = FrameworkInjector("torch", "InjectedOp", inject_variant)
  new_module = wrapper.visit(transformer)

  code = new_module.code
  assert '"InjectedOp": StandardMap(api="target.op"' in code

  # Check for dictionary presence
  assert 'inject_args={"epsilon": 1e-05, "flag": True}' in code or 'inject_args={"flag": True, "epsilon": 1e-05}' in code


def test_injector_missed_target():
  """
  Verify behavior when the target framework class is not found.
  """
  source = """ 
@register_framework("jax") 
class Adapter: 
    pass
"""
  wrapper = cst.parse_module(source)
  # Looking for 'torch', but only 'jax' exists
  transformer = FrameworkInjector("torch", "Op", FrameworkVariant(api="foo"))
  wrapper.visit(transformer)

  assert transformer.found is False


def test_injector_handles_empty_args(target_source):
  """
  Verify standard map construction when optional fields are missing.
  Ensures syntax like `StandardMap(api="...")` with no trailing comma/args.
  """
  simple_var = FrameworkVariant(api="simple.api")
  wrapper = cst.parse_module(target_source)
  transformer = FrameworkInjector("torch", "SimpleOp", simple_var)
  new_module = wrapper.visit(transformer)

  code = new_module.code

  # Strict matching without spaces around equals due to tight_eq usage
  assert '"SimpleOp": StandardMap(api="simple.api")' in code
  assert "args=" not in code
  assert "requires_plugin=" not in code


def test_syntax_validity(target_source, sample_variant):
  """
  Ensure generated code is valid Python syntax by compiling it.
  """
  wrapper = cst.parse_module(target_source)
  transformer = FrameworkInjector("torch", "Op", sample_variant)
  new_module = wrapper.visit(transformer)

  try:
    compile(new_module.code, "<string>", "exec")
  except SyntaxError as e:
    pytest.fail(f"Generated invalid python syntax: {e}")


def test_import_injection_scipy(target_source):
  """
  Scenario: Variant uses 'scipy.special.erf'.
  Existing source does NOT import scipy.
  Expectation: 'import scipy' inserted at top.
  """
  # Input source does not have scipy
  scipy_var = FrameworkVariant(api="scipy.special.erf")
  wrapper = cst.parse_module(target_source)
  transformer = FrameworkInjector("torch", "Erf", scipy_var)
  new_module = wrapper.visit(transformer)

  code = new_module.code

  # Check logic injection
  assert '"Erf": StandardMap(api="scipy.special.erf")' in code
  # Check import injection
  assert "import scipy" in code
  # Ensure it is at the top
  assert code.find("import scipy") < code.find("@register_framework")


def test_import_injection_skip_existing(target_source):
  """
  Scenario: Source already has 'import scipy'.
  Expectation: No double import.
  """
  # Helper: Inject scipy manually at top
  prepped_source = "import scipy\n" + target_source
  scipy_var = FrameworkVariant(api="scipy.special.erf")

  wrapper = cst.parse_module(prepped_source)
  transformer = FrameworkInjector("torch", "Erf", scipy_var)
  new_module = wrapper.visit(transformer)

  code = new_module.code
  # Should have 1 occurence (from source)
  assert code.count("import scipy") == 1


def test_import_injection_handles_docstring_header(sample_variant):
  """
  Scenario: Module starts with docstring and future.
  """
  # Note: sample_variant is 'torch...', so we expect 'import torch' injection
  # BUT torch is probably already there. Let's make a new variant.
  new_var = FrameworkVariant(api="extra_lib.func")

  source = '"""Header Doc."""\nfrom __future__ import annotations\n\nclass Adapter: pass'

  wrapper = cst.parse_module(source)
  transformer = FrameworkInjector("target", "Op", new_var)
  new_module = wrapper.visit(transformer)

  code = new_module.code

  assert "import extra_lib" in code
  # Ensure constraints
  # Docstring first
  assert code.find('"""Header Doc."""') == 0
  # Future second
  assert code.find("from __future__") > code.find('"""')
  # Import third
  assert code.find("import extra_lib") > code.find("from __future__")
