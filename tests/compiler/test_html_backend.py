"""Tests for the HTML Compiler Backend."""

from ml_switcheroo.core.compiler.backends.html import HtmlBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode


def test_html_backend_init():
  """Test HTML backend initialization."""
  backend = HtmlBackend()
  assert backend is not None


def test_html_backend_format_args():
  """Test format_args method."""
  backend = HtmlBackend()
  metadata = {"arg_1": "val1", "other": "val2", "arg_3": "val3"}
  formatted = backend._format_args(metadata)
  assert formatted == "val1, other=val2, val3"


def test_html_backend_is_stateful():
  """Test is_stateful method."""
  backend = HtmlBackend()

  node_input = LogicalNode(id="in", kind="Input")
  assert backend._is_stateful(node_input) is False

  node_output = LogicalNode(id="out", kind="Output")
  assert backend._is_stateful(node_output) is False

  node_func_id = LogicalNode(id="func_1", kind="Linear")
  assert backend._is_stateful(node_func_id) is False

  node_func_kind = LogicalNode(id="n1", kind="func_call")
  assert backend._is_stateful(node_func_kind) is False

  node_upper = LogicalNode(id="n2", kind="Linear")
  assert backend._is_stateful(node_upper) is True

  node_lower = LogicalNode(id="n3", kind="relu")
  assert backend._is_stateful(node_lower) is False


def test_html_backend_clean_kind():
  """Test clean_kind method."""
  backend = HtmlBackend()
  assert backend._clean_kind("func_call") == "Call"
  assert backend._clean_kind("a.b.c.Relu") == "Relu"
  assert backend._clean_kind("relu") == "Relu"


def test_html_backend_create_arrow():
  """Test create_arrow method."""
  backend = HtmlBackend()

  arrow_def = backend._create_arrow(1, 2, "def")
  assert arrow_def.style_class == "s-red"
  assert arrow_def.x2 == 60
  assert arrow_def.y2 == 80

  arrow_data = backend._create_arrow(1, 2, "data")
  assert arrow_data.style_class == "s-green"
  assert arrow_data.x2 == 60
  assert arrow_data.y2 == 0

  arrow_seq = backend._create_arrow(1, 3, "seq")
  assert arrow_seq.style_class == "s-blue"
  assert arrow_seq.y2 == 50 + (3 - 1 - 1) * 120  # 3 - 1 = 2 -> 2 - 1 = 1 -> 50 + 1 * 120 = 170

  arrow_unknown = backend._create_arrow(1, 2, "unknown")
  assert arrow_unknown.x2 == 0
  assert arrow_unknown.y2 == 0


def test_html_backend_layout_graph_empty():
  """Test layout_graph with empty graph or only inputs/outputs."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="test")
  graph.nodes.append(LogicalNode(id="in", kind="Input"))
  graph.nodes.append(LogicalNode(id="out", kind="Output"))

  boxes = backend._layout_graph(graph)
  assert boxes == []


def test_html_backend_layout_graph_nodes():
  """Test layout_graph with stateful and stateless nodes."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="test_graph")

  # Add nodes to graph
  n1 = LogicalNode(id="n1", kind="Linear", metadata={"arg_x": "1"})  # Stateful
  n2 = LogicalNode(id="n2", kind="relu", metadata={"alpha": "0.1"})  # Stateless
  n3 = LogicalNode(id="n3", kind="Linear")  # Stateful

  graph.nodes.extend([n1, n2, n3])
  # Add some dummy edges to satisfy topological sort if it cares (it might just return nodes if no edges)
  # Actually, topological_sort of disconnected nodes will just return them in some order.

  boxes = backend._layout_graph(graph)

  # Expected boxes:
  # n1: 1 red, 1 blue, 1 green
  # n2: 1 blue, 1 green
  # n3: 1 red, 1 blue, 1 green
  # 1 return bubble
  # Total = 3 + 2 + 3 + 1 = 9 boxes
  assert len(boxes) == 9

  # Check return circle
  assert boxes[-1].css_class == "circ"


def test_html_backend_compile():
  """Test compile method."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="MyModel")
  n1 = LogicalNode(id="n1", kind="relu")
  graph.nodes.append(n1)

  html_str = backend.compile(graph)
  assert "MyModel" in html_str


def test_html_backend_compile_default_name():
  """Test compile method with default name."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="GeneratedNet")
  n1 = LogicalNode(id="n1", kind="relu")
  graph.nodes.append(n1)

  html_str = backend.compile(graph)
  assert "ConvNet" in html_str


def test_html_backend_compile_no_name():
  """Test compile method with no name."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="")
  n1 = LogicalNode(id="n1", kind="relu")
  graph.nodes.append(n1)

  html_str = backend.compile(graph)
  assert "ConvNet" in html_str


def test_html_backend_layout_graph_no_last_blue_row():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")

  n1 = LogicalNode("n1", "add")
  graph.nodes.append(n1)
  backend._layout_graph(graph)


def test_html_backend_layout_graph_break_loop():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
  from unittest.mock import patch

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.nodes.append(n1)
  graph.nodes.append(n2)

  # We will mock the GridBox creation to prevent "box b" from being in the css_class
  from ml_switcheroo.core.compiler.backends.html import GridBox

  orig_gridbox = GridBox

  def mock_gridbox(*args, **kwargs):
    """Test function."""
    css = kwargs.get("css_class", "")
    if "box b" in css:
      kwargs["css_class"] = "box mutated"
    return orig_gridbox(*args, **kwargs)

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    backend._layout_graph(graph)


def test_html_backend_layout_graph_no_match_last_blue_row2():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
  from unittest.mock import patch

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.nodes.append(n1)
  graph.nodes.append(n2)

  orig_gridbox = __import__("ml_switcheroo.core.compiler.backends.html", fromlist=["GridBox"]).GridBox

  def mock_gridbox(*args, **kwargs):
    """Test function."""
    css = kwargs.get("css_class", "")
    if "box b" in css:
      kwargs["css_class"] = "box mutated"
    return orig_gridbox(*args, **kwargs)

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    backend._layout_graph(graph)


def test_html_backend_layout_graph_no_match_last_blue_row3():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")

  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.nodes.append(n1)
  graph.nodes.append(n2)

  from unittest.mock import patch
  import ml_switcheroo.core.compiler.backends.html as html_mod

  orig_gridbox = html_mod.GridBox

  def mock_gridbox(*args, **kwargs):
    """Test function."""
    box = orig_gridbox(*args, **kwargs)
    if "box b" in box.css_class:
      box.css_class = "box g"
    return box

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    backend._layout_graph(graph)


def test_html_backend_layout_graph_no_match_last_blue_row4():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")

  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.nodes.append(n1)
  graph.nodes.append(n2)

  from unittest.mock import patch
  import ml_switcheroo.core.compiler.backends.html as html_mod

  orig_gridbox = html_mod.GridBox

  def mock_gridbox(*args, **kwargs):
    """Test function."""
    box = orig_gridbox(*args, **kwargs)
    if "box b" in box.css_class:
      box.css_class = "box g"
    return box

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    backend._layout_graph(graph)


def test_html_backend_layout_graph_no_match_last_blue_row5():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")

  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.nodes.append(n1)
  graph.nodes.append(n2)

  from unittest.mock import patch
  import ml_switcheroo.core.compiler.backends.html as html_mod

  orig_gridbox = html_mod.GridBox

  def mock_gridbox(*args, **kwargs):
    """Test function."""
    box = orig_gridbox(*args, **kwargs)
    if "box b" in box.css_class:
      box.css_class = "box g"
    return box

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    backend._layout_graph(graph)


def test_html_backend_layout_graph_no_match_last_blue_row6():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")

  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.nodes.append(n1)
  graph.nodes.append(n2)

  from unittest.mock import patch
  import ml_switcheroo.core.compiler.backends.html as html_mod

  orig_gridbox = html_mod.GridBox

  def mock_gridbox(*args, **kwargs):
    """Test function."""
    box = orig_gridbox(*args, **kwargs)
    if "box b" in box.css_class:
      box.row = 999
    return box

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    backend._layout_graph(graph)


def test_html_backend_layout_graph_no_match_last_blue_row7():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")

  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.nodes.append(n1)
  graph.nodes.append(n2)

  from unittest.mock import patch
  import ml_switcheroo.core.compiler.backends.html as html_mod

  orig_gridbox = html_mod.GridBox

  def mock_gridbox(*args, **kwargs):
    """Test function."""
    box = orig_gridbox(*args, **kwargs)
    if "box b" in box.css_class:
      box.row = 999
    return box

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    backend._layout_graph(graph)


def test_html_backend_layout_graph_no_match_last_blue_row8():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")

  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.nodes.append(n1)
  graph.nodes.append(n2)

  from unittest.mock import patch
  import ml_switcheroo.core.compiler.backends.html as html_mod

  orig_gridbox = html_mod.GridBox

  def mock_gridbox(*args, **kwargs):
    """Test function."""
    box = orig_gridbox(*args, **kwargs)
    if kwargs.get("css_class") == "box b" and kwargs.get("row") == 2:
      box.css_class = "box g"
    return box

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    backend._layout_graph(graph)


def test_html_backend_layout_graph_no_match_last_blue_row9():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.html import HtmlBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.nodes.append(n1)
  graph.nodes.append(n2)

  from unittest.mock import patch
  import ml_switcheroo.core.compiler.backends.html as html_mod

  orig_gridbox = html_mod.GridBox

  def mock_gridbox(*args, **kwargs):
    """Test function."""
    box = orig_gridbox(*args, **kwargs)
    # when creating the green box for row 2, it is added to the list.
    # when iterating over the boxes looking for row 2 and "box b",
    # if we set "box b" to have row 999 instead of 2, the search fails.
    if kwargs.get("css_class") == "box b" and kwargs.get("row") == 2:
      # don't mess with the constructor row parameter itself so it doesn't break other things maybe?
      # Wait, if we set box.row = 999 here it DOES change it.
      box.row = 999
    return box

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    backend._layout_graph(graph)
