# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(1500)

import bulletarm
import bulletarm_baselines

# -- Project information -----------------------------------------------------

project = 'BulletArm'
copyright = '2022, Colin Kohler, Dian Wang'
author = 'Colin Kohler, Dian Wang'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.intersphinx',
  'sphinx.ext.autodoc',
  'sphinx.ext.todo',
  'sphinx.ext.mathjax',
  'sphinx.ext.ifconfig',
  'sphinx.ext.viewcode',
  'sphinx.ext.githubpages',
  'sphinx.ext.napoleon',
  'sphinx_rtd_theme',
  'sphinx.ext.autosectionlabel'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Latex Elementes
latex_elemenets = {
  'papersize' : 'letterpaper',
  'pointsize' : '10pt',
  'preamble' : '',
  'figure_align' : 'htbp'
}

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
  'collapse_navigation' : False,
  'display_version' : True,
  'sticky_navigation' : True,
  'navigation_depth': 2,
  'titles_only' : False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

intershpinx_mapping = {
  'python': ('https://docs.python.org/', None),
  'numpy': ('https://docs.scipy.org/doc/numpy/', None),
  'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
  'PyTorch': ('http://pytorch.org/docs/master/', None),
}

# Don't show entire module path
add_module_names = False

def setup(app):
  app.add_css_file('custom.css')
