Pending Release Notes
=====================

Updates / New Features
----------------------

Interfaces

* Added ``SaliencyMetric`` metric interface class.

Implementations

* Added ``Entropy`` metric implementation class.

CI/CD

* Removed ``mypy`` check and dependency.

Documentation

* Updated documentation format to have ``Quick Start``, ``Tutorial``, ``Explanation``, ``How-To``, and ``Reference``
  sections

* Updated ``implementations.rst`` to use ``autosummary``

* Moved ``examples`` directory to ``docs/examples``

* Updated ``DRISE.ipynb`` and ``OcclusionSaliency.ipynb`` to be XAITK tutorials.

* Corrected Google Colab links in example notebooks

* Created ``SwappableImplementations.ipynb`` as a How To guide.

* Updated ``index.rst``, ``installation.rst``, and ``README.md``  based on ``devel-jatic``.

* Replaced ``introduction.rst``  with ``xaitk_explanation.rst`` for Explanation section of docs.

* Added ``ROADMAP.md``.

* Added autodoc entry for ``SaliencyMetric`` and ``Entropy`` classes under
  ``interfaces.rst`` and ``implementations.rst`` respectively.

* Added ``xaitk_how_to_topics.rst`` to documentation.

Fixes
-----

* Fixed ``pyright`` errors.
