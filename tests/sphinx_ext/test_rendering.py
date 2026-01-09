"""
Tests for Sphinx HTML Rendering utilities (WASM Demo).

Verifies that:
1. `render_demo_html` produces valid HTML structure.
2. The template includes all required sections (Toolbar, Editors, Tabs).
3. The Weight Script tab logic is injected.
4. **Time Travel** UI elements are present (slider, buttons, config).
5. Default Frameworks are respected.
"""

import re
from typing import Dict, List
import pytest
from ml_switcheroo.sphinx_ext.rendering import render_demo_html, _render_primary_options


@pytest.fixture
def mock_registry_data():
  hierarchy = {"torch": [], "jax": [{"key": "flax_nnx", "label": "Flax"}]}
  examples = '{"key":"val"}'
  meta = '{"torch": ["neural"]}'
  return hierarchy, examples, meta


def test_render_demo_html_structure(mock_registry_data):
  """
  Scenario: Generating the demo HTML block.
  Expectation: Contains all structural divs, inputs, Weight Script, and Time Travel toolbar.
  """
  hierarchy, ex, meta = mock_registry_data
  html = render_demo_html(hierarchy, ex, meta)

  # 1. Root Container
  assert '<div id="switcheroo-wasm-root"' in html
  assert 'class="demo-header"' in html

  # 2. Controls
  assert 'id="select-src"' in html
  assert 'id="select-tgt"' in html
  assert 'id="select-example"' in html
  assert 'id="btn-convert"' in html

  # 3. Output Tabs
  assert 'for="tab-console"' in html
  assert 'for="tab-mermaid"' in html
  assert 'for="tab-trace"' in html

  # 4. Weight Script Tab
  assert 'label for="tab-weights"' in html

  # 5. NEW: Time Travel UI
  assert 'class="time-travel-bar"' in html
  assert 'id="tt-slider"' in html
  assert 'id="btn-tt-prev"' in html
  assert 'id="btn-tt-next"' in html
  assert 'id="tt-phase-label"' in html

  # 6. NEW: Pipeline Config
  assert 'class="tt-config"' in html
  assert 'id="chk-opt-graph"' in html
  assert 'id="chk-opt-imports"' in html


def test_default_selection_logic(mock_registry_data):
  """
  Scenario: Verify default "selected" attribute injection.
  """
  hierarchy, ex, meta = mock_registry_data
  html = render_demo_html(hierarchy, ex, meta)

  # We expect 'torch' to be selected as source
  assert re.search(r'<option value="torch"[^>]*selected', html)

  # Expect 'jax' to be selected as target
  assert re.search(r'<option value="jax"[^>]*selected', html)


def test_flavour_dropdown_rendering():
  """
  Scenario: Rendering flavour secondary selector.
  """
  hierarchy = {"jax": [{"key": "flax_nnx", "label": "Flax"}, {"key": "haiku", "label": "Haiku"}]}

  # We call internal helper processing used in render_demo_html implicitly via import
  from ml_switcheroo.sphinx_ext.rendering import _render_flavour_dropdown

  html = _render_flavour_dropdown("tgt", hierarchy, "jax")

  assert 'id="tgt-flavour"' in html
  assert '<option value="flax_nnx" selected>Flax</option>' in html
  assert '<option value="haiku" >Haiku</option>' in html
