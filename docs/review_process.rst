Review Process
**************

The purpose of this document is to define the process for reviewing and
integrating branches into XAITK-Saliency.

See ``CONTRIBUTING.md`` for guidelines on contributing to XAITK-Saliency.

See `release process`_ for guidelines on the release process for XAITK-Saliency.

.. _`release process`: release_process.html

.. contents:: The review process consists of the following steps:

Pull Request
============
A PR is initiated by a user intending to integrate a branch from their forked
repository.
Before the branch is integrated into the XAITK-Saliency master branch, it must
first go through a series of checks and a review to ensure that the branch is
consistent with the rest of the repository and doesn't contain any issues.

Workflow Status
---------------
The submitter must set the status of their PR.

Draft
^^^^^
Indicates that the submitter does not think that the PR is in a reviewable or
mergeable state.
Once they complete their work and think that the PR is ready to be considered
for merger, they may set the status to ``Open``.

Open
^^^^
Indicates that a PR is ready for review and that the submitter of the PR thinks
that the branch is ready to be merged.
If the submitter is still working on the PR and simply wants feedback, they
must request it and leave their branch marked as a ``Draft``.

Closed
^^^^^^
Indicates that the PR is resolved or discarded.


Continuous Integration
======================
The following checks are included in the automated portion of the review
process.
These are run as part of the CI/CD pipeline driven by GitHub actions.
The success or failure of each may be seen in-line in a submitted PR in the
"Checks" section of the PR view.

ghostflow
---------
Runs basic checks on the commits submitted in a PR.
Should ghostflow find any issues with the build, the diagnostics are written
out prompting the submitter to correct the reported issues.
If there are no issues, or just warnings, ghostflow simply reports a successful
build.
The branch should usually pass this check before it can be merged.
In the rare case that a PR is not subject to a change note, then failure of
this check specifically in regard to that lower-level check, may be ignored by
the reviewer.

Some reports such as whitespace issues will need to be corrected by rewriting
the commit.
This is generally handled by performing ``git commit --fixup=...`` commits and
performing a ``git rebase -i --autosquash ...`` rebase afterwards, or a more
simple global squash if appropriate.

LGTM Analysis
-------------
Runs a more advanced code analysis tool over the branch that can address issues
that the submitter might not have noticed.
Should LGTM find an issue, it will write a comment on the PR and provide a link
to the LGTM website for more detailed information.
The comment should be addressed by the submitter before continuing to submit
for review.
Ideally a submitted PR adds no new issues as reported by LGTM.

Passage of this check is not strictly required but highly encouraged.

SonarCloud Code Analysis
------------------------
Similar to LGTM, this service performs a more in-depth analysis of the code.
This should provide the same output as a SonarQube scan.

Passage of this check is not strictly required but highly encouraged.

lint
----
Runs ``flake8`` to quality check the code style.
You can run this check manually in your local repository with
``poetry run flake8``.

Passage of this check is strictly required.

MyPy
----
Performs static type analysis.
You can run this check manually in your local repository with ``poetry run
mypy``.

Passage of this check is strictly required.

Unittests
---------
Runs the unittests created under ``tests/`` as well as any doc-tests found in
doc-strings in the package code proper.
You can run this check manually  in your local repository with ``poetry run
pytest``.

Passage of these checks is strictly required.

CodeCov
-------
This check reports aggregate code coverage as reported from output of the
unittest jobs.
This check requires that all test code be "covered" (i.e. there is no dead-code
in the tests) and that a minimum coverage bar is met for package code changed
or added in the PR.
The configuration for this may be found in the ``codecov.yml`` file in the
repository root.

Passage of these checks is strictly required.

ReadTheDocs Documentation Build
-------------------------------
This check ensures that the documentation portion of the package is buildable
by the current host ReadTheDocs.org.

Passage of these checks is strictly required.

Example Notebooks Execution
---------------------------
This check executes included example notebooks to ensure their proper
functionality with the package with respect to a pull request.
Not all notebooks may be run as some maybe set up to use too many resources or
run for an extended period of time.


Human Review
============
Once the automatic checks are either resolved or addressed, the submitted PR
will need to go through a human review.
Reviewers should add comments to provide feedback and raise potential issues.
Should the PR pass their review, the reviewer should then indicate that it has
their approval using the Github review interface to flag the PR as ``Approved``.

A review can still be requested before the checks are resolved, but the PR must
be marked as a ``Draft``.
Once the PR is in a mergeable state, it will need to undergo a final review to
ensure that there are no outstanding issues.

If a PR is not a draft and has an approving review, it may be merged at any
time.

Notebooks
---------
The default preference is that all Jupyter Notebooks be included in execution
of the Notebook CI workflow (here: ``.github/workflows/ci-example-notebooks.yml``).
If a notebook is added, it should be verified that it has been added to the
list of notebooks to be run.
If it has not been, the addition should be requested or for a rationale as to
why it has not been.
Rationale for specific notebooks should be added to the relevant section in
``examples/README.md``.

Resolving a Branch
==================

Merge
-----
Once a PR receives an approving review and is no longer marked as a ``Draft``,
the repository maintainers can merge it, closing the pull request.
It is recommended that the submitter delete their branch after the PR is
merged.

Close
-----
If it is decided that the PR will not be integrated into XAITK-Saliency, then
it can be closed through Github.
