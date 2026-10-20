# docs/conf.py

# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

# add the project root (one level up) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# nbsphinx uses pandoc for Markdown cells. The documentation requirements
# provide a portable pandoc binary; register it when available so Read the
# Docs and local builds use the same converter.
try:
    import pypandoc

    pandoc_directory = os.path.dirname(pypandoc.get_pandoc_path())
    os.environ["PATH"] = pandoc_directory + os.pathsep + os.environ["PATH"]
except (ImportError, OSError):
    pass

# -- Project information -----------------------------------------------------
project = 'PVCracks'
authors = [
    "Norman Jost",
    "Brandon K. Byford",
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
    release = '0.2.0'
    version = '0.2'

# -- General configuration ---------------------------------------------------

# Sphinx extensions
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.viewcode',
  'sphinx.ext.napoleon',
  'sphinx.ext.mathjax',
  'nbsphinx',
]
# optional: never execute the notebooks during build
nbsphinx_execute = 'never'

# Serve the exact executed notebooks shown in these pages as .ipynb downloads.
# Sphinx resolves the leading slash relative to the documentation source root.
nbsphinx_prolog = r"""
.. only:: html

   :download:`Download this notebook </{{ env.docname }}.ipynb>`

"""

# Where to look for templates
templates_path = ['_templates']

# The master toctree document
master_doc = 'index'

# Support both reStructuredText and Markdown
source_suffix = {
    '.rst': 'restructuredtext',
}

# Patterns to ignore when looking for source files.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    # Historical placeholder API pages.  The implemented modules are linked
    # explicitly from index.rst.
    'api/modules.rst',
    'api/pvcracks.rst',
    'api/pvcracks.MultiSolSegment.formatting.rst',
    'api/pvcracks.MultiSolSegment.testing.rst',
    'api/pvcracks.MultiSolSegment.training.rst',
    # These are one-off data-migration scripts that execute filesystem work
    # against user-specific paths during import, not callable package APIs.
    'api/pvcracks.utils.dataset_operations.rst',
]

# -- Options for HTML output -------------------------------------------------

# Use the Read the Docs theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    # Retain the project name with the logo.  The custom layout places the
    # release immediately beneath it and adds the repository control.
    'logo_only': False,
    'navigation_depth': 4,
}

html_context = {
    "display_version": True,
    "current_version": release,
}

# Path to static files (css, javascript, images)
# Keep custom static assets separate from documentation sources and build
# output. Including the docs root here makes Sphinx recursively copy _build.
html_static_path = ['_static']

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

# Use Python when notebook metadata does not select a lexer. Saved executions
# may declare ``ipython3``, supplied by docs' ipython-pygments-lexers dependency.
nbsphinx_codecell_lexer = 'python'

# The documented runtime APIs import scientific dependencies. Mocking them
# keeps Read the Docs API pages inspectable without downloading GPU packages.
autodoc_mock_imports = [
    'cv2',
    'imutils',
    'matplotlib',
    'skimage',
    'sklearn',
    'scipy',
    'termcolor',
    'torch',
    'torchvision',
    'tqdm',
]
