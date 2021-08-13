Pending Release Notes
=====================


Updates / New Features
----------------------

Documentation

* Added some review process documentation.

* Add initial FAQ documentation file.

* Add background material for saliency maps to ``introduction.rst``.

Implementations

* Add ``DRISEScoring`` implementation of the ``GenerateDetectorProposalSaliency``
  interface using detection output and associated occlusion masks.

Tests

* Removed use of `unittest.TestCase` as it is not utilized directly in any way
  that PyTest does not provide.


Fixes
-----
