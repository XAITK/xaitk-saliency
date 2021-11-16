Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Updated notebooks CI workflow to include notebook data caching.

Documentation

* Added text discussing black box methods to ``introduction.rst``.

* Added a section to ``introduction.rst`` that describes the links between saliency algorithms and implementations.

* Edited all text.

* Update top-level ``README.md`` file to have more useful content.

Examples

* Add example notebook for saliency on Atari deep RL agent

* Updated examples to all use a common data sub-directory when downloading or
  saving generated data.

Fixes
-----

Build

* Fix incorrect specification of actually-optional `papermill` in relation to
  its intended inclusion in the `example_deps` extra.

* Update patch version of Pillow transitive dependency locked in the
  ``poetry.lock`` file to address CVE-2021-23437.

Examples

* Updated example Jupyter notebooks with more consistent dependency checks and
  also fixed minor header formatting issues.

Tests

* Fix deprecation warnings around the use of ``numpy.random.random_integers``.

Utilities

* Fix ``xaitk_saliency.utils.detection.format_detection`` to not upcast the
  data type output when ``objectness is None``.

* Fix ``xaitk_saliency.utils.masking.weight_regions_by_scalar`` to not upcast
  the data type output when ``inv_masks is True``.
