Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Updated codecov action version to 3.

* Added explicit use of codecov token to facilitate successful coverage
  submission.

Dependencies

* Updated ``notebook`` dependency due to a vulnerability alert.

* Periodic update of locked dep versions within abstract version constraints.

* Updated sphinx versions to fix local documentation building issue.

* Updated some abstract dependencies to be more future-open for latest valid
  package version compatibility.

Fixes
-----

Docs

* Added missing step to the release process about creating the release on
  GitHub's Releases section.

Examples

* Added a note to each example about restarting the runtime for compatibility
  with Colab, as well as a step to create a data directory if necessary.
