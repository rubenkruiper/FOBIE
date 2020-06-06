# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath("../SORE"))
# sys.path.insert(0, os.path.abspath("../SORE/my_utils"))
import recommonmark
from sphinx.domains import Domain

class GithubURLDomain(Domain):
    """
    Resolve certain links in markdown files to github source.
    """

    name = "githuburl"
    ROOT = "https://github.com/tensorpack/tensorpack/blob/master/"

    def resolve_any_xref(self, env, fromdocname, builder, target, node, contnode):
        github_url = None
        if ".html" not in target:
            if target.startswith("../../") and not target.startswith("../../modules"):
                url = target.replace("../", "")
                github_url = url

        if github_url is not None:
            if github_url.endswith("README"):
                # bug of recommonmark.
                # https://github.com/readthedocs/recommonmark/blob/ddd56e7717e9745f11300059e4268e204138a6b1/recommonmark/parser.py#L152-L155
                github_url += ".md"
            print("Ref {} resolved to github:{}".format(target, github_url))
            contnode["refuri"] = self.ROOT + github_url
            return [("githuburl:any", contnode)]
        else:
            return []



# -- Project information -----------------------------------------------------

project = 'Semi-Open Relation Extraction (SORE)'
copyright = '2020, Ruben Kruiper'
author = 'Ruben Kruiper'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.viewcode',
              'recommonmark']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


### Added markdown support
numpydoc_show_class_members = False
# source_parsers = {'.md': CommonMarkParser}
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# At the bottom of conf.py

def setup(app):
    from recommonmark.transform import AutoStructify
    app.add_config_value('recommonmark_config', {
            'enable_auto_toc_tree': True,
            }, True)
    app.add_transform(AutoStructify)
    app.add_domain(GithubURLDomain)