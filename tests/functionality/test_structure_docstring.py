"""
Tests for Docstring Argument Updater (Feature 4).

Verifies that the `StructureMixin` updates the function docstring
when arguments are injected (e.g. `rngs`).

Covered Scenarios:
1.  **Google Style**: Append to `Args:` section.
2.  **NumPy Style**: Append to `Parameters` section.
3.  **Missing Section**: Create `Args:` block.
4.  **No Duplicate**: Do not append if argument is mentioned.
5.  **Single/Double Quotes**: Robust extraction.
"""

import pytest
import libcst as cst
from typing import Tuple, Optional, List

from ml_switcheroo.core.rewriter.structure import StructureMixin
from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockRewriter(StructureMixin, BaseRewriter):
  """Test wrapper for StructureMixin logic."""

  pass


@pytest.fixture
def rewriter():
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  # Use bare semantics
  mgr.data = {}
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return MockRewriter(mgr, config)


def apply_doc_update(rewriter, code: str, injected_args: List[Tuple[str, Optional[str]]]) -> str:
  """
  Parses code, extracts the top-level function, runs _update_docstring, returns generated code.
  """
  module = cst.parse_module(code)
  func_def = module.body[0]

  if not isinstance(func_def, cst.FunctionDef):
    raise ValueError("Test code must start with a function definition")

  new_func = rewriter._update_docstring(func_def, injected_args)

  # Re-wrap in module to generate code
  new_module = module.with_changes(body=[new_func])
  return new_module.code


def test_google_style_injection(rewriter):
  """Verify appending to Args section."""
  code = ''' 
def forward(self, x): 
    """ 
    Forward Pass. 

    Args: 
        x: Input tensor. 
    """ 
    return x
'''
  injected = [("rngs", "Rngs")]
  result = apply_doc_update(rewriter, code, injected)

  assert "rngs: Injected state argument." in result
  assert "x: Input tensor." in result
  # Should be inside Args
  assert result.find("Args:") < result.find("rngs:")


def test_numpy_style_injection(rewriter):
  """Verify appending to Parameters section (with underlines)."""
  code = ''' 
def forward(self, x): 
    """ 
    Forward Pass. 

    Parameters 
    ---------- 
    x : array 
        Input. 
    """ 
    pass
'''
  injected = [("key", None)]
  result = apply_doc_update(rewriter, code, injected)

  assert "key: Injected state argument." in result
  assert "Parameters" in result
  assert result.find("Parameters") < result.find("key:")


def test_missing_section_creation(rewriter):
  """Verify creating Args section if missing."""
  code = ''' 
def forward(self, x): 
    """ 
    Simple Docstring. 
    """ 
    pass
'''
  injected = [("rngs", None)]
  result = apply_doc_update(rewriter, code, injected)

  assert "Args:" in result
  assert "rngs: Injected state argument." in result


def test_duplicate_prevention(rewriter):
  """Verify we don't inject if argument is already documented."""
  code = ''' 
def forward(self, x): 
    """ 
    Args: 
        rngs: Pre-existing doc. 
    """ 
    pass
'''
  injected = [("rngs", "Rngs")]
  result = apply_doc_update(rewriter, code, injected)

  # Should assume existing doc covers it, or at least not duplicate
  assert result.count("rngs:") == 1
  assert "Injected state argument." not in result


def test_single_quote_docstring(rewriter):
  """Verify handling of single quotes."""
  code = "def f(x):\n    'Single line doc.'\n    pass"
  injected = [("rng", None)]
  result = apply_doc_update(rewriter, code, injected)

  # Implementation handles single quotes by injecting newlines
  assert "rng: Injected state argument." in result
  assert "Single line doc." in result


def test_raw_string_handling(rewriter):
  """Verify r-strings."""
  code = ''' 
def f(x): 
    r""" 
    Regex doc. 
    args: 
       x: val 
    """ 
    pass
'''
  injected = [("rngs", None)]
  result = apply_doc_update(rewriter, code, injected)

  assert 'r"""' in result
  assert "rngs: Injected state argument." in result
