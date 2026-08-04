"""Test suite for the Rendering module."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.sphinx_ext.rendering import render_demo_html, _render_primary_options, _render_flavour_dropdown


@patch("ml_switcheroo.sphinx_ext.rendering.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.rendering.get_adapter")
def test_render_primary_options(mock_get_adapter, mock_priority):
  """Renders primary options."""
  mock_priority.return_value = ["torch", "jax", "unknown"]
  hierarchy = {"torch": [], "jax": [], "unknown": []}
  mock_get_adapter.return_value = MagicMock(display_name="Mock")
  html = _render_primary_options(hierarchy)
  assert 'value="torch"' in html
  assert 'value="jax"' in html
  assert 'value="unknown"' in html


def test_render_flavour_dropdown():
  """Renders flavour dropdown."""
  hierarchy = {"jax": [{"key": "flax_nnx", "label": "Flax"}], "torch": []}
  html = _render_flavour_dropdown("src", hierarchy, "jax")
  assert 'value="flax_nnx"' in html

  html_empty = _render_flavour_dropdown("src", {"torch": []}, "torch")
  assert "No Flavours" in html_empty


@patch("ml_switcheroo.sphinx_ext.rendering.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.rendering._render_primary_options")
@patch("ml_switcheroo.sphinx_ext.rendering._render_flavour_dropdown")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.glob")
@patch("os.path.getmtime")
def test_render_demo_html(mock_mtime, mock_glob, mock_exists, mock_flavours, mock_primary, mock_priority):
  """Renders demo HTML."""
  mock_priority.return_value = ["torch", "jax"]
  mock_primary.return_value = '<option value="torch">Torch</option><option value="jax">Jax</option>'
  mock_flavours.return_value = "<select></select>"
  mock_exists.return_value = True
  wheel_mock = MagicMock()
  wheel_mock.name = "test.whl"
  mock_glob.return_value = [wheel_mock]
  mock_mtime.return_value = 1
  hierarchy = {"torch": [], "jax": []}
  html = render_demo_html(hierarchy, "{}", "{}")
  assert "test.whl" in html
  assert "torch" in html
  assert "jax" in html


@patch("ml_switcheroo.sphinx_ext.rendering.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.rendering._render_primary_options")
@patch("ml_switcheroo.sphinx_ext.rendering._render_flavour_dropdown")
@patch("pathlib.Path.exists")
def test_render_demo_html_no_priority_fallback(mock_exists, mock_flavours, mock_primary, mock_priority):
  """Renders demo HTML with empty priority fallback."""
  mock_exists.return_value = False
  mock_priority.return_value = []
  mock_primary.return_value = '<option value="source_placeholder">Placeholder</option>'
  mock_flavours.return_value = ""
  # custom doesn't trigger torch or jax shortcuts
  html = render_demo_html({"custom": []}, "{}", "{}")
  assert "source_placeholder" in html


@patch("ml_switcheroo.sphinx_ext.rendering.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.rendering._render_primary_options")
@patch("ml_switcheroo.sphinx_ext.rendering._render_flavour_dropdown")
@patch("pathlib.Path.exists")
def test_render_demo_html_no_dist(mock_exists, mock_flavours, mock_primary, mock_priority):
  """Renders demo HTML no dist."""
  mock_exists.return_value = False
  mock_priority.return_value = ["torch", "jax"]
  mock_primary.return_value = ""
  mock_flavours.return_value = ""
  html = render_demo_html({"torch": []}, "{}", "{}")
  assert "ml_switcheroo-latest-py3-none-any.whl" in html


@patch("ml_switcheroo.sphinx_ext.rendering.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.rendering._render_primary_options")
@patch("ml_switcheroo.sphinx_ext.rendering._render_flavour_dropdown")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.glob")
def test_render_demo_html_no_wheels(mock_glob, mock_exists, mock_flavours, mock_primary, mock_priority):
  """Renders demo HTML no wheels."""
  mock_exists.return_value = True
  mock_glob.return_value = []
  mock_priority.return_value = ["torch", "jax"]
  mock_primary.return_value = ""
  mock_flavours.return_value = ""
  html = render_demo_html({"torch": []}, "{}", "{}")
  assert "ml_switcheroo-latest-py3-none-any.whl" in html
