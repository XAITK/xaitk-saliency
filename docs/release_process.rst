Steps of the xaitk-saliency Release Process
===========================================
Three types of releases are expected to occur:
  - major
  - minor
  - patch

See the ``CONTRIBUTING.md`` file for information on how to contribute features
and patches.

The following process should apply when any release that changes the version
number occurs.

Create and Merge Version Update Branch
--------------------------------------

Major and Minor Releases
^^^^^^^^^^^^^^^^^^^^^^^^
Major and minor releases may add one or more trivial or non-trivial features
and functionalities.

1. Create a new branch off of the ``master`` named something like
   ``update-to-v{NEW_VERSION}``, where ``NEW_VERSION`` is the new ``X.Y``
   version.

   a. Use the ``scripts/update_release_notes.sh`` script to update the project
      version number, create ``docs/release_notes/v{NEW_VERSION}.rst``, and add
      a new pending release notes stub file.

      .. code-block:: bash

         $ # When creating a major release
         $ ./scripts/update_release_notes.sh major
         $ # OR when creating a minor release
         $ ./scripts/update_release_notes.sh minor

   b. Add a descriptive paragraph under the title section of
      ``docs/release_notes/v{NEW_VERSION}.rst`` summarizing this release.

2. Push the created branch to the upstream repository, not your fork (this is
   an exception to the normal forking workflow).

3. Create a pull/merge request for this branch with ``release`` as the merge
   target. This is to ensure that everything passes CI testing before making
   the release. If there is an issue, then topic branches should be made and
   merged into this branch until the issue is resolved.

4. Get an approving review.

5. Merge the pull/merge request into the ``release`` branch.

6. Tag the resulting merge commit.
   See `Tag new version`_ below for how to do this.

7. As a repository administrator, merge the ``release`` branch into ``master``
   locally and push the updated ``master`` to upstream. (Replace "upstream"
   in the example below with your applicable remote name.)

   .. code-block:: bash

      $ git fetch --all
      $ git checkout upstream/master
      $ git merge --log --no-ff upstream/release
      $ git push upstream master

7. Update version reference in the `XAITK/xaitk.github.io home page`_ to the
   new version.

Patch Release
^^^^^^^^^^^^^
A patch release should only contain fixes for bugs or issues with an existing
release.
No new features or functionality should be introduced in a patch release.
As such, patch releases should only ever be based on an existing release point
(git tag).

This list assumes we are creating a new patch release off of the *latest*
release version, i.e. off of the ``release`` branch.
If a patch release for an older release version is being created, see the
`Patching an Older Release`_ section.

1. Create a new branch off of the ``release`` branch named something like
   ``update-to-v{NEW_VERSION}``, where ``NEW_VERSION`` is the target ``X.Y.Z``,
   including the bump in the patch (``Z``) version component.

   a. Use the ``scripts/update_release_notes.sh`` script to update the project
      version number, create ``docs/release_notes/v{NEW_VERSION}.rst``, and add
      a new pending release notes stub file.

      .. code-block:: bash

         $ ./scripts/update_release_notes.sh patch

   b. Add a descriptive paragraph under the title section of
      ``docs/release_notes/v{NEW_VERSION}.rst`` summarizing this release.

2. Push the created branch to the upstream repository, not your fork (this is
   an exception to the normal forking workflow).

3. Create a pull/merge request for this branch with ``release`` as the merge
   target. This is to ensure that everything passes CI testing before making
   the release. If there is an issue, then topic branches should be made and
   merged into this branch until the issue is resolved.

4. Get an approving review.

5. Merge the pull/merge request into the ``release`` branch.

6. Tag the resulting merge commit.
   See `Tag new version`_ below for how to do this.

7. As a repository administrator, merge the ``release`` branch into ``master``
   locally and push the updated ``master`` to upstream. (Replace "upstream"
   in the example below with your applicable remote name.)

   .. code-block:: bash

      $ git fetch --all
      $ git checkout upstream/master
      $ git merge --log --no-ff upstream/release
      $ git push upstream master

8. If this patch release now represents the highest version of the package,
   update version reference in the `XAITK/xaitk.github.io home page`_ to the
   new version.

Patching an Older Release
"""""""""""""""""""""""""
When patching a major/minor release that is not the latest pair, a branch needs
to be created based on the release version being patched to integrate the
specific patches into.
This branch should be prefixed with ``release-`` to denote that it is a release
integration branch.
Patch topic-branches should be based on this branch.
When all fix branches have been integrated, follow the `Patch Release`_ section
above, replacing ``release`` branch references (merge target) to be the
``release-...`` integration branch.
Step 6 should be to merge this release integration branch into ``release``
first, and *then* ``release`` into ``master``, if applicable (some patches may
only make sense for specific versions).

Tag new version
---------------
Release branches are tagged in order to record where in the git tree a
particular release refers to.
All release tags should be in the history of the ``release`` and ``master``
branches (barring exceptional circumstances).

We prefer to use local ``git tag`` commands to create the release version
tag, pushing the tag to upstream.
The version tag should be applied to the merge commit resulting from the
above described ``update-to-v{NEW_VERSION}`` topic-branch ("the release").

See the example commands below, replacing ``HASH`` with the appropriate git
commit hash, and ``UPSTREAM`` with the appropriate remote name.
We also show how to use `Poetry's version command`_ to consistently access the
current package version.

.. code-block:: bash

   $ git checkout HASH
   # VERSION="v$(poetry version -s)"
   $ git tag -a "$VERSION" -F docs/release_notes/"$VERSION".rst
   $ git push UPSTREAM "$VERSION"

After creating and pushing a new version tag, a GitHub "release" should be
made.
Navigate to the `releases page on GitHub`_ and click the ``Draft a new
release`` button in the upper right.
The newly added tag should be selected in the "Choose a tag" drop-down.
The "Release Title" should be the version tag (i.e. "v#.#.#").
Copy and paste this version's release notes into the ``Describe this release``
text box.
Remember to check the ``This is a pre-release`` check-box if appropriate.
Click the ``Public release`` button at the bottom of the page when complete.


.. _Poetry's version command: https://python-poetry.org/docs/cli/#version
.. _releases page on GitHub: https://github.com/XAITK/xaitk-saliency/releases
.. _XAITK/xaitk.github.io home page: https://github.com/XAITK/xaitk.github.io/edit/master/_pages/home.md#L12
