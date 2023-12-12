# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'marl-factory-grid'
copyright = '2023, Steffen Illium, Robert Mueller, Joel Friedrich'
author = 'Steffen Illium, Robert Mueller, Joel Friedrich'
release = '2.5.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [#'myst_parser',
                  'sphinx.ext.todo',
                  'sphinx.ext.autodoc',
                  'sphinx.ext.intersphinx',
                  # 'sphinx.ext.autosummary',
                  'sphinx.ext.linkcode',
                  'sphinx_mdinclude',
              ]

templates_path = ['_templates']
exclude_patterns = ['marl_factory_grid.utils.proto', 'marl_factory_grid.utils.proto.fiksProto_pb2*']


autoclass_content = 'both'
autodoc_class_signature = 'separated'
autodoc_typehints = 'description'
autodoc_inherit_docstrings = True
autodoc_typehints_format = 'short'
autodoc_default_options = {
    'members': True,
    # 'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    # 'exclude-members': '__weakref__',
    'show-inheritance': True,
}
autosummary_generate = True
add_module_names = False
toc_object_entries = False
modindex_common_prefix = ['marl_factory_grid.']

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
from pathlib import Path
import sys
sys.path.insert(0, (Path(__file__).parents[2]).resolve().as_posix())
sys.path.insert(0, (Path(__file__).parents[2] / 'marl_factory_grid').resolve().as_posix())

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_book_theme"  # 'alabaster'
# html_static_path = ['_static']

# In your configuration, you need to specify a linkcode_resolve function that returns an URL based on the object.
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html


def linkcode_resolve(domain, info):
    if domain in ['py', '__init__.py']:
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/illiumst/marl-factory-grid/%s.py" % filename

print(sys.executable)
