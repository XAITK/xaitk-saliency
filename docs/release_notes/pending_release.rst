Pending Release Notes
=====================

Updates / New Features
----------------------

Build

* New minimum supported python changed to `python = "^3.8"`.

CI

* Updated codecov action version to 3.

* Added explicit use of codecov token to facilitate successful coverage
  submission.

Dependencies

* Updated ``notebook`` dependency due to a vulnerability alert.

* Periodic update of locked dep versions within abstract version constraints.

* Updated sphinx versions to fix local documentation building issue.

* Updated python minimum requirement to 3.8 (up from 3.6). This involved a
  number of updates and bifurcations of abstract requirements, an update to
  pinned versions for development/CI, and expansion of CI to cover python
  versions 3.10 and 3.11 (latest current release).

Fixes
-----

Docs

* Added missing step to the release process about creating the release on
  GitHub's Releases section.

Examples

* Added a note to each example about restarting the runtime for compatibility
  with Colab, as well as a step to create a data directory if necessary.
