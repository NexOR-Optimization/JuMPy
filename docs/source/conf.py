from __future__ import annotations

from pathlib import Path

project = "JuMPy"
author = "JuMPy contributors"
copyright = "2026, JuMPy contributors"
html_baseurl = "https://nexor-optimization.github.io/JuMPy/"

root_dir = Path(__file__).resolve().parents[2]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
nitpicky = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/NexOR-Optimization/JuMPy",
    "use_repository_button": True,
    "use_issues_button": True,
}

sphinx_gallery_conf = {
    "examples_dirs": str(root_dir / "docs" / "tutorials"),
    "gallery_dirs": "generated/tutorials",
    "filename_pattern": r".*\.py",
    "abort_on_example_error": True,
    "only_warn_on_example_error": False,
    "remove_config_comments": True,
    "show_memory": False,
}
