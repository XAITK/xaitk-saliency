Steps of the xaitk-saliency Release Process
===========================================
Three types of releases are expected to occur:

* patch
* minor
* major

See the ``CONTRIBUTING.md`` file for information on how to contribute features
and patches.

The following process should apply when any release that changes the version
number occurs.

Create and Merge Version Update Branch
--------------------------------------

Patch Release
^^^^^^^^^^^^^
A patch release should only contain fixes for bugs or issues with an existing
release.
No new features or functionality should be introduced in a patch release.
As such, patch releases should only ever be based on an existing release point.

1. Create a new branch off of the appropriate ``vX.Y.Z`` tag named something
   like ``release-patch-{NEW_VERSION}``, where ``NEW_VERSION`` is an increase
   in the ``Z`` version component.

   a. Use ``poetry version patch`` to increase the patch value appropriately in
      the :file:`pyproject.toml` file.

   b. Rename the ``docs/release_notes/pending_patch.rst`` file to
      ``docs/release_notes/v{VERSION}.rst``, matching the new version value.
      Add a descriptive paragraph under the title section summarizing this
      release.

   c. Add a reference to the new release notes RST file in
      ``docs/release_notes.rst``.

   d. In a separate commit, add back a blank pending release notes file stub.
      See `Stub Pending Notes File`_.

2. Create a pull/merge request for this branch with master as the merge target.
   This is to ensure that everything passes CI testing before making the
   release. If there is an issue, then branches should be made and merged into
   this branch until the issue is resolved.

3. Tag branch (see `Tag new version`_ below ) after resolving issues and before
   merging into ``master``.

4. Merge version bump branch into ``master`` branch.

5. `Create new version release to PYPI`_.

6. Update version reference in the `XAITK/xaitk.github.io home page`_ to the
   new version.

Major and Minor Releases
^^^^^^^^^^^^^^^^^^^^^^^^
Major and minor releases may add one or more trivial or non-trivial features
and functionalities.

1. Create a new branch off of the ``master`` named something like
   ``release-[major,minor]-{NEW_VERSION}``.

   a. Increment patch value in  ``pyproject.toml`` file's ``version`` attribute
      under the `[tool.poetry]` section.

      * See `Poetry's version command`_ for a convenient means of incrementing
        the version.

   b. Rename the ``docs/release_notes/pending_release.rst`` file to
      ``docs/release_notes/v{VERSION}.rst``, matching the new version value.
      Add a descriptive paragraph under the title section summarizing this
      release.

   c. Add a reference to the new release notes RST file in
      ``docs/release_notes.rst``.

   d. In a separate commit, add back a blank pending release notes file stub.
      See `Stub Pending Notes File`_.

2. Create a pull/merge request for this branch with master as the merge target.
   This is to ensure that everything passes CI testing before making the
   release. If there is an issue, then branches should be made and merged into
   this branch until the issue is resolved.

3. Tag branch (see `Tag new version`_ below) after resolving issues and before
   merging into ``master``.

4. Merge version bump branch into the ``master`` branch.

5. `Create new version release to PYPI`_.

6. Update version reference in the `XAITK/xaitk.github.io home page`_ to the
   new version.

Stub Pending Notes File
^^^^^^^^^^^^^^^^^^^^^^^
The following is the basic content that goes into the stub pending release
notes file:

.. code-block::

    Pending Release Notes
    =====================

    Updates / New Features
    ----------------------

    Fixes
    -----

Tag New Version
---------------
Release branches should be tagged in order to record where in the git tree a
particular release refers to.
The branch off of ``master`` is usually the target of such tags.

Currently the ``From GitHub`` method is preferred as it creates a "verified"
release.

From GitHub
^^^^^^^^^^^
Navigate to the `releases page on GitHub`_ and click the ``Draft a new
release`` button in the upper right.

Fill in the new version in the ``Tag version`` text box (e.g. ``v#.#.#``)
and use the same string in the ``Release title`` text box.
The "@" target should be the release branch created above.

Copy and paste this version's release notes into the ``Describe this release``
text box.

Remember to check the ``This is a pre-release`` check-box if appropriate.

Click the ``Public release`` button at the bottom of the page when complete.

From Git on the Command Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a new git tag using the new version number (format:
``v<MAJOR.<MINOR>.<PATCH>``) on the merge commit for the version update branch
merger::

    $ git tag -a -m "[Major|Minor|Patch] release v#.#.#"

Push this new tag to GitHub (or appropriate remote)::

    $ git push origin v#.#.#

To add the release notes to GitHub, navigate to the `tags page on GitHub`_
and click on the "Add release notes" link for the new release tag.  Copy and
paste this version's release notes into the description field and the version
number should be used as the release title.

Create new version release to PYPI
----------------------------------

__ https://python-poetry.org/docs/repositories/#configuring-credentials

We will use Poetry again to perform package building and publishing.
See `this documentation`__ on how to set and store your PYPA credentials in Poetry.

Make sure the source is checked out on the appropriate  version tag, the repo
is clean (no uncommited files/edits). ``git clean`` may help ensure a clean
state::

    $ git checkout <VERSION_TAG>
    $ git clean -xdi  # NOTE: `-i` makes this an interactive command.

Build source and wheel packages for the current version::

    $ poetry build

The files in `./dist/` may be inspected for correctness before publishing to
PYPA with::

    $ poetry publish


.. _Poetry's version command: https://python-poetry.org/docs/cli/#version
.. _releases page on GitHub: https://github.com/XAITK/xaitk-saliency/releases
.. _tags page on GitHub: https://github.com/XAITK/xaitk-saliency/tags
.. _XAITK/xaitk.github.io home page: https://github.com/XAITK/xaitk.github.io/edit/master/_pages/home.md#L12
