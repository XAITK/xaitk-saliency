Installation
============

There are two ways to obtain the xaitk-saliency package.
The simplest is to install via the :command:`pip` command.
Alternatively, you can use `Poetry`_ (`installation`_ and `usage`_) to acquire the source tree and
develop locally.


.. _installation: Poetry-installation_
.. _usage: Poetry-usage_


From :command:`pip`
-------------------

.. prompt:: bash

    pip install xaitk-saliency

This method will install all of the same functionality as when installing from source.
If you have an existing installation and would like to upgrade your version,
provide the ``-U``/``--upgrade`` `option`__.

__ Pip-install-upgrade_


From Source
-----------
The following assumes `Poetry`_ is already installed.

Quick Start
^^^^^^^^^^^

.. prompt:: bash

    cd /where/things/should/go/
    git clone https://github.com/XAITK/xaitk-saliency.git ./
    poetry install
    poetry run pytest
    cd docs
    poetry run make html


Installing Python Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This project uses `Poetry`_ for dependency management, environment consistency,
package building, version management, and publishing to PyPI.
Dependencies are `abstractly defined`_ in the :file:`pyproject.toml` file, as
well as `specifically pinned versions`_ in the :file:`poetry.lock` file, both
of which can be found in the root of the source tree.

.. _abstractly defined: Poetry-dependencies_
.. _specifically pinned versions: Poetry-poetrylock_

The following installs both installation and development dependencies as
specified in the :file:`pyproject.toml` file, with versions specified
(including for transitive dependencies) in the :file:`poetry.lock` file:

.. prompt:: bash

    poetry install --sync --with linting,tests,docs


Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^
The documentation for xaitk-saliency is maintained as a collection of
`reStructuredText`_ documents in the :file:`docs/` folder of the project.
The :program:`Sphinx` documentation tool can process this documentation
into a variety of formats, the most common of which is HTML.

Within the :file:`docs/` directory is a Unix :file:`Makefile` (for Windows
systems, a :file:`make.bat` file with similar capabilities exists).
This :file:`Makefile` takes care of the work required to run :program:`Sphinx`
to convert the raw documentation to an attractive output format.
For example, as shown in the Quick Start section (above), calling ``make html`` will generate
HTML format documentation rooted at :file:`docs/_build/html/index.html`.

Calling the command ``make help`` here will show the other documentation
formats that may be available (although be aware that some of them require
additional dependencies such as :program:`TeX` or :program:`LaTeX`).


Live Preview
""""""""""""

While writing documentation in a markup format such as `reStructuredText`_, it
is very helpful to preview the formatted version of the text.
While it is possible to simply run the ``make html`` command periodically, a
more seamless workflow of this is available.
Within the :file:`docs/` directory is a small Python script called
:file:`sphinx_server.py` that can simply be called with:

.. prompt:: bash

    poetry run python sphinx_server.py

This will run a small process that watches the :file:`docs/` folder contents,
as well as the source files in :file:`src/xaitk_saliency/`, for changes.
:command:`make html` is re-run automatically when changes are detected.
This will serve the resulting HTML files at http://localhost:5500.
Having this URL open in a browser will provide you with an up-to-date
preview of the rendered documentation.


.. _Pip-install-upgrade: https://pip.pypa.io/en/stable/reference/pip_install/#cmdoption-U
.. _Poetry: https://python-poetry.org
.. _Poetry-installation: https://python-poetry.org/docs/#installation
.. _Poetry-usage: https://python-poetry.org/docs/basic-usage/
.. _Poetry-poetrylock: https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock
.. _Poetry-dependencies: https://python-poetry.org/docs/pyproject/#dependencies-and-dev-dependencies
.. _Sphinx: http://sphinx-doc.org/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
