import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------
project = 'ml-switcheroo'
author = 'ml-switcheroo Developers'
copyright = f'{datetime.now().year}, {author}'
version = '0.0.1'
release = version

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../src'))

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
    'myst_parser',
    'sphinxcontrib.mermaid',    # <--- Ensure this is enabled
    'sphinx_copybutton',
    'sphinx_material',
]

# -- AutoAPI Configuration ---------------------------------------------------
autoapi_dirs = ['../src']
autoapi_type = 'python'
autoapi_root = 'api'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]
autoapi_ignore = ['*test*', '*migrations*']

# -- Mermaid Configuration ---------------------------------------------------
# Optional: Explicitly set the CDN version if rendering fails locally
mermaid_version = "10.6.1"

# -- MyST Parser Configuration -----------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "tasklist",
]
myst_heading_anchors = 3

# VITAL: This tells MyST to convert ```mermaid blocks into the \
# '.. mermaid::' directive provided by sphinxcontrib-mermaid
myst_fence_as_directive = ["mermaid"]

# -- Theme Configuration -----------------------------------------------------
html_theme = 'sphinx_material'
html_title = 'ml-switcheroo'
html_short_title = 'switcheroo'

html_theme_options = {
    'nav_title': 'ml-switcheroo',
    'color_primary': 'blue',
    'color_accent': 'light-blue',
    'repo_url': 'https://github.com/yourusername/ml-switcheroo',
    'repo_name': 'ml-switcheroo',
    'globaltoc_depth': 2,
    'globaltoc_collapse': True,
    'globaltoc_includehidden': True,
    'master_doc': False,
    'version_dropdown': False,
    'table_classes': ['plain'],
}

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

html_static_path = ['_static']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
