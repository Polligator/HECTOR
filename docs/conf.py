"""Sphinx configuration for the HECTOR documentation site.

The tutorials are the notebooks in ``examples/``, rendered with their saved
outputs. They are not re-executed here: running them needs the hosted model
checkpoints and, for reasonable speed, an accelerator. Re-run a notebook
locally and commit it to update its page.
"""

import pathlib
import shutil

DOCS = pathlib.Path(__file__).parent
EXAMPLES = DOCS.parent / "examples"
TUTORIALS = DOCS / "tutorials"

# Sphinx only reads sources under docs/, so mirror the notebooks and the images
# they link to. The copy is a build product and is ignored by git.
shutil.rmtree(TUTORIALS, ignore_errors=True)
shutil.copytree(EXAMPLES, TUTORIALS, ignore=shutil.ignore_patterns(".ipynb_checkpoints"))

project = "HECTOR"
html_title = "HECTOR"

extensions = ["myst_nb"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

nb_execution_mode = "off"
nb_merge_streams = True
myst_enable_extensions = ["colon_fence"]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["site.js"]
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = False
