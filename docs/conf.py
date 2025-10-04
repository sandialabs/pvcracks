# docs/conf.py

# -- Path setup --------------------------------------------------------------
import os
import sys

# add the project root (one level up) to sys.path so that
# `import pvcracks` works
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'PVCracks'
authors = [
    "Norman Jost",
    "Ojas Sanghi",
    "Emma Cooper",
    "Jennifer L. Braid"
]

# If pvcracks defines __version__, you can import it:
try:
    import pvcracks
    release = pvcracks.__version__
    version = release
except ImportError:
    # fallback if not installed yet
    release = '0.1.0'
    version = '0.1.0'

# -- General configuration ---------------------------------------------------

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',    # auto-generate docs from docstrings
    'sphinx.ext.viewcode',   # add links to highlighted source
    'sphinx.ext.napoleon',   # support Google/Numpy style docstrings
    'sphinx.ext.mathjax',    # render math in HTML
]

# Where to look for templates
templates_path = ['_templates']

# The master toctree document
master_doc = 'index'

# # Support both reStructuredText and Markdown
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.md': 'markdown',
# }

# Patterns to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# Use the Read the Docs theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}

# Path to static files (css, javascript, images)
# Create docs/_static/ and copy your logos in there.
html_static_path = ['.']

# Logo and favicon
html_logo = 'pvcracks_logo.png'
html_favicon = 'duramat_logo.png'

# -- Autodoc options ---------------------------------------------------------

# Show both the class docstring and the __init__ docstring
autoclass_content = 'both'

# If your docstrings use typing annotations, you may need this:
# autoclass_content = 'init'
# autodata_content = 'both'

# -- Napoleon settings (if you use Google or NumPy style docstrings) ---------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
