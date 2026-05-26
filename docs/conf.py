project = "pyens"
copyright = "2025, pyens contributors"
author = "pyens contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

html_theme = "pydata_sphinx_theme"
html_title = "pyens"

html_theme_options = {
    "github_url": "https://github.com/TARPS-group/pyens",
    "header_links_before_dropdown": 4,
    "navbar_end": ["navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
}

html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

suppress_warnings = ["myst.header", "myst.xref_missing"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
