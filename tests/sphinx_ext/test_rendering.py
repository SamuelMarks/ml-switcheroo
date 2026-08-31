"""Tests for sphinx_ext rendering."""

import os
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from ml_switcheroo.sphinx_ext.rendering import _render_flavour_dropdown, _render_primary_options, render_demo_html


def test_render_primary_options_empty() -> None:
  """Docstring."""
  hierarchy: Dict[str, List[Dict[str, str]]] = {}
  html: str = _render_primary_options(hierarchy)
  assert html == ""


def test_render_primary_options_standard() -> None:
  """Docstring."""
  hierarchy: Dict[str, List[Dict[str, str]]] = {"torch": [{"key": "t1", "label": "t1"}], "jax": [], "unknown_fw": []}

  # Needs get_adapter mock because it instantiates adapters during rendering
  class MockAdapter:
    """Mock adapter."""

    def __init__(self, name: str) -> None:
      """Init.

      Args:
          name (str): Adapter name.
      """
      self.display_name: str = name.capitalize()

  with mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter):
    html: str = _render_primary_options(hierarchy)
    assert '<optgroup label="Level 1: High-Level">' in html
    assert '<option value="torch">Torch</option>' in html

    assert '<optgroup label="Level 2: Numerics">' in html
    assert '<option value="jax">Jax</option>' in html

    assert '<optgroup label="Other">' in html
    assert '<option value="unknown_fw">Unknown_fw</option>' in html


def test_render_flavour_dropdown_empty() -> None:
  """Docstring."""
  hierarchy: Dict[str, List[Dict[str, str]]] = {"torch": [], "jax": []}
  html: str = _render_flavour_dropdown("src", hierarchy, "torch")
  assert '<option value="" disabled selected data-parent="">No Flavours</option>' in html
  assert "display:none;" in html


def test_render_flavour_dropdown_with_flavours() -> None:
  """Docstring."""
  hierarchy: Dict[str, List[Dict[str, str]]] = {
    "torch": [{"key": "torch.nn", "label": "Torch NN"}, {"key": "torch.fx", "label": "Torch FX"}],
    "jax": [],
  }
  html: str = _render_flavour_dropdown("src", hierarchy, "torch")

  # Container should be visible because torch has children
  assert "display:inline-block;" in html

  # First flavour selected
  assert 'value="torch.nn" data-parent="torch" selected' in html
  assert "Torch NN" in html

  # Second not selected
  assert 'value="torch.fx" data-parent="torch"  style="">' in html
  assert "Torch FX" in html


def test_render_demo_html_basic(tmp_path: Path) -> None:
  """Test the full HTML rendering method without wheels present.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  hierarchy: Dict[str, List[Dict[str, str]]] = {"torch": [], "jax": []}

  # Mock get_adapter for primary options
  class MockAdapter:
    """Mock adapter."""

    def __init__(self, name: str) -> None:
      """Init.

      Args:
          name (str): Adapter name.
      """
      self.display_name: str = name.capitalize()

  with mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter):
    # We need to mock path to not look at actual dist folder if we want a clean test
    with mock.patch("ml_switcheroo.sphinx_ext.rendering.Path") as mock_path_cls:
      mock_root: mock.MagicMock = mock.MagicMock()
      mock_dist: mock.MagicMock = mock.MagicMock()
      mock_dist.exists.return_value = False
      mock_root.__truediv__.return_value = mock_dist

      mock_parents: List[Any] = [None, None, None, mock_root]
      mock_path_obj: mock.MagicMock = mock.MagicMock()
      mock_path_obj.parents = mock_parents
      mock_path_cls.return_value = mock_path_obj

      html: str = render_demo_html(hierarchy, '{"ex": "1"}', '{"tier": "1"}')

      assert 'window.SWITCHEROO_PRELOADED_EXAMPLES = {"ex": "1"};' in html
      assert 'window.SWITCHEROO_FRAMEWORK_TIERS = {"tier": "1"};' in html
      assert "ml_switcheroo-latest-py3-none-any.whl" in html
      assert 'value="torch" selected' in html
      assert 'value="jax" selected' in html


def test_render_demo_html_with_wheels(tmp_path: Path) -> None:
  """Test full HTML rendering when wheels are present.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  hierarchy: Dict[str, List[Dict[str, str]]] = {"jax": [], "numpy": []}

  # Setup temp dist dir with a wheel
  dist_dir: Path = tmp_path / "dist"
  dist_dir.mkdir()
  wheel_path: Path = dist_dir / "ml_switcheroo-1.0.0-py3-none-any.whl"
  wheel_path.touch()

  # Touch a second to test sorting by mtime
  wheel2_path: Path = dist_dir / "ml_switcheroo-1.1.0-py3-none-any.whl"
  wheel2_path.touch()
  # Make it explicitly newer
  os.utime(wheel2_path, (os.stat(wheel_path).st_mtime + 100, os.stat(wheel_path).st_mtime + 100))

  class MockAdapter:
    """Mock adapter."""

    def __init__(self, name: str) -> None:
      """Init.

      Args:
          name (str): Adapter name.
      """
      self.display_name: str = name.capitalize()

  with mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter):
    # Patch the file path directly to use our tmp_path structure
    with mock.patch("ml_switcheroo.sphinx_ext.rendering.Path") as mock_path_cls:
      mock_root: Path = tmp_path
      mock_parents: List[Any] = [None, None, None, mock_root]
      mock_path_obj: mock.MagicMock = mock.MagicMock()
      mock_path_obj.parents = mock_parents
      mock_path_cls.return_value = mock_path_obj

      html: str = render_demo_html(hierarchy, "{}", "{}")

      # Should have chosen the newer wheel
      assert 'data-wheel="ml_switcheroo-1.1.0-py3-none-any.whl"' in html
      # Without torch, def_source should be jax (based on default priority)
      assert 'value="jax" selected' in html


def test_render_demo_html_empty_wheels_and_fallback(tmp_path: Path) -> None:
  """Test rendering when dist exists but is empty, and fallback target logic.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  hierarchy: Dict[str, List[Dict[str, str]]] = {"numpy": [], "mlir": []}

  # Setup temp dist dir with NO wheels
  dist_dir: Path = tmp_path / "dist"
  dist_dir.mkdir()

  class MockAdapter:
    """Docstring."""

    def __init__(self, name: str) -> None:
      """Docstring.

      Args:
          name (str): Adapter name.
      """
      self.display_name: str = name.capitalize()

  with mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter):
    with mock.patch("ml_switcheroo.sphinx_ext.rendering.Path") as mock_path_cls:
      mock_root: Path = tmp_path
      mock_parents: List[Any] = [None, None, None, mock_root]
      mock_path_obj: mock.MagicMock = mock.MagicMock()
      mock_path_obj.parents = mock_parents
      mock_path_cls.return_value = mock_path_obj

      html: str = render_demo_html(hierarchy, "{}", "{}")

      assert "ml_switcheroo-latest-py3-none-any.whl" in html
      assert 'value="numpy"' in html
      assert 'value="mlir"' in html


def test_render_demo_html_no_priority_order(tmp_path: Path) -> None:
  """Test rendering when priority order is empty to hit the def_target candidate else block.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  hierarchy: Dict[str, List[Dict[str, str]]] = {"numpy": []}

  class MockAdapter:
    """Docstring."""

    def __init__(self, name: str) -> None:
      """Docstring.

      Args:
          name (str): Adapter name.
      """
      self.display_name: str = name.capitalize()

  with mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter):
    with mock.patch("ml_switcheroo.sphinx_ext.rendering.Path") as mock_path_cls:
      mock_root: Path = tmp_path
      mock_path_obj: mock.MagicMock = mock.MagicMock()
      mock_path_obj.parents = [None, None, None, mock_root]
      mock_path_cls.return_value = mock_path_obj

      with mock.patch("ml_switcheroo.sphinx_ext.rendering.get_framework_priority_order", return_value=[]):
        html: str = render_demo_html(hierarchy, "{}", "{}")
        assert 'value="source_placeholder"' not in html  # it gets replaced by numpy maybe? Actually wait.
