"""
Tests for the Standards Injector (LibCST Transformer).
"""

import pytest
import libcst as cst
from ml_switcheroo.core.dsl import OperationDef, ParameterDef, FrameworkVariant
from ml_switcheroo.tools.injector_spec import StandardsInjector


@pytest.fixture
def sample_op():
  """Returns a valid OperationDef."""
  return OperationDef(
    operation="LogSoftmax",
    description="Log Softmax implementation.",
    std_args=[ParameterDef(name="input", type="Tensor"), ParameterDef(name="dim", type="int", default="-1")],
    variants={"torch": FrameworkVariant(api="torch.log_softmax")},
  )


def test_injector_finds_internal_ops(sample_op):
  """Verify it locates INTERNAL_OPS and inserts the new key."""
  source_code = "INTERNAL_OPS = {}"

  wrapper = cst.MetadataWrapper(cst.parse_module(source_code))
  transformer = StandardsInjector(sample_op)
  new_module = wrapper.visit(transformer)

  assert transformer.found is True

  code = new_module.code
  assert '"LogSoftmax":' in code
  assert '"description": "Log Softmax implementation."' in code
  assert '"std_args":' in code
  # Verify rich param structure with strict double quotes from json.dumps
  assert '"name": "input"' in code
  assert '"type": "Tensor"' in code
  assert '"default": "-1"' in code


def test_injector_ignores_other_assigns(sample_op):
  """Verify it ignores irrelevant assignments."""
  source_code = "OTHER_DICT = {}"

  wrapper = cst.parse_module(source_code)
  transformer = StandardsInjector(sample_op)
  wrapper.visit(transformer)

  assert transformer.found is False


def test_injector_appends_to_existing(sample_op):
  """Verify it appends to a non-empty dictionary correctly."""
  source_code = """ 
INTERNAL_OPS = { 
    'Existing': {'description': 'old'} 
} 
"""
  wrapper = cst.parse_module(source_code)
  transformer = StandardsInjector(sample_op)
  new_module = wrapper.visit(transformer)

  code = new_module.code
  assert "'Existing':" in code
  assert '"LogSoftmax":' in code

  # Check syntax validity by compiling
  try:
    compile(code, "<string>", "exec")
  except SyntaxError:
    pytest.fail(f"Generated Invalid Python code:\n{code}")
