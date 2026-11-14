"""Tests for sphinx_ext rendering."""

import os
from pathlib import Path
from unittest import mock

from ml_switcheroo.sphinx_ext.rendering import _render_flavour_dropdown, _render_primary_options, render_demo_html


def test_render_primary_options_empty() -> None:
  """Verify `_render_primary_options` handles empty as expected."""
  hierarchy: dict[str, list[dict[str, str]]] = {}
  html: str = _render_primary_options(hierarchy)
  assert html == ""


def test_render_primary_options_standard() -> None:
  """Verify `_render_primary_options` handles standard as expected."""
  hierarchy: dict[str, list[dict[str, str]]] = {
    "torch": [{"key": "t1", "label": "t1"}],
    "jax": [],
    "unknown_fw": [],
  }

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


def test_render_primary_options_ir_grouped() -> None:
  """Verifies that ir and ml_switcheroo_ir are grouped under Level 3: Standard IR with explicit labels."""
  hierarchy: dict[str, list[dict[str, str]]] = {
    "ir": [],
    "ml_switcheroo_ir": [],
    "torch": [],
  }

  class MockAdapter:
    """Mock adapter returning distinct IR display names."""

    def __init__(self, name: str) -> None:
      """Init adapter.

      Args:
          name (str): Adapter framework key.
      """
      if name == "ir":
        self.display_name: str = "ML-Switcheroo IR (JSON)"
      elif name == "ml_switcheroo_ir":
        self.display_name = "ML-Switcheroo IR (Python)"
      else:
        self.display_name = name.capitalize()

  with mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter):
    html: str = _render_primary_options(hierarchy)
    assert '<optgroup label="Level 3: Standard IR">' in html
    assert '<option value="ir">ML-Switcheroo IR (JSON)</option>' in html
    assert '<option value="ml_switcheroo_ir">ML-Switcheroo IR (Python)</option>' in html
    assert "Level 3: Standard IR" in html


def test_render_flavour_dropdown_empty() -> None:
  """Verify `_render_flavour_dropdown` emits `display:none;` for empty."""
  hierarchy: dict[str, list[dict[str, str]]] = {"torch": [], "jax": []}
  html: str = _render_flavour_dropdown("src", hierarchy, "torch")
  assert '<option value="" disabled selected data-parent="">No Flavours</option>' in html
  assert "display:none;" in html


def test_render_flavour_dropdown_with_flavours() -> None:
  """Verify `_render_flavour_dropdown` emits `display:inline-block;` for with flavours."""
  hierarchy: dict[str, list[dict[str, str]]] = {
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
  hierarchy: dict[str, list[dict[str, str]]] = {"torch": [], "jax": []}

  # Mock get_adapter for primary options
  class MockAdapter:
    """Mock adapter."""

    def __init__(self, name: str) -> None:
      """Init.

      Args:
          name (str): Adapter name.
      """
      self.display_name: str = name.capitalize()

  # We need to mock path to not look at actual dist folder if we want a clean test
  with (
    mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter),
    mock.patch("ml_switcheroo.sphinx_ext.rendering.Path") as mock_path_cls,
  ):
    mock_root: mock.MagicMock = mock.MagicMock()
    mock_dist: mock.MagicMock = mock.MagicMock()
    mock_dist.exists.return_value = False
    mock_root.__truediv__.return_value = mock_dist

    mock_parents = [None, None, None, mock_root]
    mock_path_obj: mock.MagicMock = mock.MagicMock()
    mock_path_obj.parents = mock_parents
    mock_path_cls.return_value = mock_path_obj

    html: str = render_demo_html(hierarchy, '{"ex": "1"}', '{"tier": "1"}')

    assert 'window.SWITCHEROO_PRELOADED_EXAMPLES = {"ex": "1"};' in html
    assert 'window.SWITCHEROO_FRAMEWORK_TIERS = {"tier": "1"};' in html
    assert "ml_switcheroo-latest-py3-none-any.whl" in html
    assert 'value="torch" selected' in html
    assert 'value="jax" selected' in html
    assert 'id="chk-on-device-exec"' in html
    assert 'id="on-device-config-bar"' in html
    assert 'id="select-conversion-mode"' in html
    assert '<option value="preserve_trivia" selected>' in html
    assert '<option value="optimized">' in html
    assert 'id="select-opt-level"' in html
    assert 'id="select-exec-backend"' in html
    assert '<option value="webgpu" selected>' in html
    assert '<option value="cuda">' in html
    assert '<option value="metal">' in html
    assert 'id="select-input-preset"' in html
    assert 'id="input-tensor-shape"' in html
    assert 'id="select-input-dtype"' in html
    assert 'id="input-random-seed"' in html
    assert 'id="select-loss-reduction"' in html
    assert 'id="chk-inject-annotations"' in html
    assert 'id="chk-inject-assertions"' in html
    assert 'id="chk-emit-bwd-code"' in html
    assert 'id="tab-runtime"' in html
    assert 'id="panel-runtime"' in html
    assert 'id="runtime-summary-badges"' in html
    assert 'id="runtime-impasse-card"' in html
    assert 'id="runtime-telemetry-table"' in html
    assert 'id="runtime-param-grads-table"' in html
    assert 'id="runtime-ir-inspector"' in html
    assert 'id="select-ir-inspector-view"' in html
    assert 'id="runtime-ir-inspector-code"' in html


def test_render_demo_html_with_wheels(tmp_path: Path) -> None:
  """Test full HTML rendering when wheels are present.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  hierarchy: dict[str, list[dict[str, str]]] = {"jax": [], "numpy": []}

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

  # Patch the file path directly to use our tmp_path structure
  with (
    mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter),
    mock.patch("ml_switcheroo.sphinx_ext.rendering.Path") as mock_path_cls,
  ):
    mock_root: Path = tmp_path
    mock_parents = [None, None, None, mock_root]
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
  hierarchy: dict[str, list[dict[str, str]]] = {"numpy": [], "mlir": []}

  # Setup temp dist dir with NO wheels
  dist_dir: Path = tmp_path / "dist"
  dist_dir.mkdir()

  class MockAdapter:
    """Sentinel `MockAdapter` fixture class used in rendering tests."""

    def __init__(self, name: str) -> None:
      """Initialize mock adapter with capitalized display name.

      Args:
          name (str): Adapter name.
      """
      self.display_name: str = name.capitalize()

  with (
    mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter),
    mock.patch("ml_switcheroo.sphinx_ext.rendering.Path") as mock_path_cls,
  ):
    mock_root: Path = tmp_path
    mock_parents = [None, None, None, mock_root]
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
  hierarchy: dict[str, list[dict[str, str]]] = {"numpy": []}

  class MockAdapter:
    """Sentinel `MockAdapter` fixture class used in rendering tests."""

    def __init__(self, name: str) -> None:
      """Initialize mock adapter with capitalized display name.

      Args:
          name (str): Adapter name.
      """
      self.display_name: str = name.capitalize()

  with (
    mock.patch("ml_switcheroo.sphinx_ext.rendering.get_adapter", side_effect=MockAdapter),
    mock.patch("ml_switcheroo.sphinx_ext.rendering.Path") as mock_path_cls,
  ):
    mock_root: Path = tmp_path
    mock_path_obj: mock.MagicMock = mock.MagicMock()
    mock_path_obj.parents = [None, None, None, mock_root]
    mock_path_cls.return_value = mock_path_obj

    with mock.patch("ml_switcheroo.sphinx_ext.rendering.get_framework_priority_order", return_value=[]):
      html: str = render_demo_html(hierarchy, "{}", "{}")
      assert 'value="source_placeholder"' not in html  # it gets replaced by numpy maybe? Actually wait.
