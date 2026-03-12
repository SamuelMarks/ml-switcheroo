"""
Documentation Page Renderer.

This module converts the dictionary View Model (produced by `DocContextBuilder`)
into final ReStructuredText (RST) content. It embeds custom HTML/CSS/JS logic to
create an interactive "Vertical Tabs" layout for displaying framework variants side-by-side.
"""

import textwrap
from typing import Any, Dict


class OpPageRenderer:
  """
  Renders RST/HTML for a single Operation documentation page.
  """

  def render_rst(self, context: Dict[str, Any]) -> str:
    """
    Generates the full .rst content for the operation.

    Args:
        context: View Model dict containing name, description, args, and variants.

    Returns:
        str: ReStructuredText source string.
    """
    op_name = context["name"]
    desc = context["description"]
    args = context["args"]
    variants = context.get("variants", [])

    # 1. Header Block
    underline = "=" * len(op_name)
    rst = [
      f"{op_name}",
      underline,
      "",
      desc,
      "",
    ]

    # 2. Signature Block
    if args:
      rst.append("**Abstract Signature:**")
      rst.append("")
      rst.append(f"``{op_name}({', '.join(args)})``")
      rst.append("")

    # 3. Variants (Interactive HTML Block)
    if variants:
      rst.append(".. raw:: html")
      rst.append("")
      # Must indent HTML content for the raw directive
      html_block = self._render_html_tabs(variants)
      indented_html = textwrap.indent(html_block, "    ")
      rst.append(indented_html)
      rst.append("")
    else:
      rst.append("*No implementations mapped.*")

    return "\n".join(rst)

  def _render_html_tabs(self, variants: list) -> str:
    """
    Generates the HTML structure for the vertical tabs UI.

    Structure:
      <div class="op-tabs-container">
        <div class="op-tabs-nav">
           <button...>Framework Name</button>
        </div>
        <div class="op-tabs-content">
           <div class="op-tab-pane">Details...</div>
        </div>
      </div>

    Args:
        variants: List ofvariant dictionaries.

    Returns:
        str: HTML block string.
    """
    nav_buttons = []
    panes = []

    for idx, v in enumerate(variants):
      is_active = "active" if idx == 0 else ""
      fw_label = v["framework"]
      api = v["api"]
      impl_type = v["implementation_type"]
      link = v["doc_url"]

      # Build Button
      btn = f'<button class="op-tab-btn {is_active}" onclick="openOpTab(event, \'{fw_label}_{idx}\')">{fw_label}</button>'
      nav_buttons.append(btn)

      # Build Pane Content
      details = []
      details.append(f"<h4>{fw_label}</h4>")

      # API String
      details.append(f'<div class="op-detail-row"><span class="label">API:</span> <code>{api}</code></div>')

      # Implementation Type
      details.append(f'<div class="op-detail-row"><span class="label">Strategy:</span> <span>{impl_type}</span></div>')

      # Link
      if link:
        details.append(
          f'<div class="op-detail-row"><a href="{link}" target="_blank" class="op-doc-link">Official Docs ↗</a></div>'
        )

      # Add more debugging details if needed (raw_def keys)
      # e.g. arguments mapping table could go here in v2

      pane = f'<div id="{fw_label}_{idx}" class="op-tab-pane {is_active}">{"".join(details)}</div>'
      panes.append(pane)

    return f""" 
<div class="op-tabs-container"> 
  <div class="op-tabs-nav"> 
    {"".join(nav_buttons)} 
  </div> 
  <div class="op-tabs-content"> 
    {"".join(panes)} 
  </div> 

  <!-- Load JS Logic only once per page ideally, but safe to exist globally --> 
  <script src="../_static/op_tabs.js"></script> 
</div> 
"""
