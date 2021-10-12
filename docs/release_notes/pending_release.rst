Pending Release Notes
=====================


Updates / New Features
----------------------

CI

* Added workflow for test running *some* example notebooks.

Documentation

* Added text discussing white box methods to ``introduction.rst``.

* Added some review process documentation.

* Add initial FAQ documentation file.

* Add background material for saliency maps to ``introduction.rst``.

* Added API docs section, which includes descriptions of all interfaces.

* Added content to the ``CONTRIBUTING.md`` file on:

  * including notes here for added updates, features and fixes

  * Jupyter notebook CI workflow inclusion

* Add implementations section.

* Update example Jupyter notebooks to work with Google Colab.

Examples

* Add example notebook using classifier-based interfaces and implementations
  with scikit-learn on the MNIST dataset.

Implementations

* Add ``DRISEScoring`` implementation of the ``GenerateDetectorProposalSaliency``
  interface using detection output and associated occlusion masks.

Tests

* Removed use of ``unittest.TestCase`` as it is not utilized directly in any way
  that PyTest does not provide.

Utilities

* Add type annotation, documentation and unit-tests for using image matrices as
  the fill option instead of just a solid color.

* Add ``format_detection`` helper function to form the input for
  ``GenerateDetectorProposalSaliency`` from separated components.

Fixes
-----

Implementations

* Fixed ``ValueError`` messages raised in the ``SimilarityScoring``
  implementation. Added unittests to check the raising and message content.
