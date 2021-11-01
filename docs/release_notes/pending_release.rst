Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Updated notebooks CI workflow to include notebook data caching.

Documentation

* Added text discussing black box methods to ``introduction.rst``.

* Added a section to ``introduction.rst`` that describes the links between sgit aliency algorithms and implementations.

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
