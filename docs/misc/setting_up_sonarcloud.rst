Setting Up xaitk-saliency with SonarCloud
=========================================

Create a SonarCloud Organization
----------------------------------
This will house our repository ("project") dashboard and is parallel to the
GitHub organization.

* Go to `SonarCloud`_

* Click the "+" button in the upper right near user drop-down.

* Select "Create a new organization".

* Follow instructions.

  * Part of this will be "Installing" the SonarCloud app at the organization level.
    Select either "All repositories" or select repositories to enable SonarCloud access to.
    Currently we have selected access only to the `xaitk-saliency`_ repository.

Create a Project for xaitk-saliency
------------------------------------
* Go to `SonarCloud`_.

* Click the "+" button in the upper right near user drop-down.

* Select the "XAITK" organization created above.

* Check our repo in the repos listed.

By default this will enable automatic scanning for common situations including
PR submissions and master-branch updates.
This is currently acceptable however this does not report coverage information
due to the SonarCloud automatic scanning not having access to the unit test
coverage reports generated during the GitHub Actions-based CI workflow.
In the future we may want to switch to performing the SonarQube scanning action
inside our CI workflow, however currently there is the hurdle where PRs from
forks do not have the appropriate access to submit scan reports without learning
private security tokens.
The following section is provided as purely experimental/historical information.

Set Analysis Method to GitHub Actions
-----------------------------------------
**NOTE:** *THIS IS NOT THE CURRENT CONFIGURATION.*
This documentation here is notionally provided as it was written during
experimentation and maybe has future value if we attempted again.

This is less setting something up but more deactivating the default setting
to use SonarCloud Automatic Analysis.
We chose to do this as SonarCloud automatic analysis has no ability to consider
unittest code coverage, as this is only a byproduct of the CI unittests.

* Go to project page (e.g. https://sonarcloud.io/dashboard?id=XAITK_xaitk-saliency)

* Administration --> Analysis Method

* Toggle off "SonarCloud Automatic Analysis"

We "enabled" the GitHub action method by explicitly configuring a job in our
GitHub CI workflow to use the `SonarCloud GitHub Action`_ to run the scanner
and submit the report.

.. code-block:: yaml

    jobs:

      # Add the following step to the end of the `unittests` job.
      unittests:
        steps:
          - name: Upload test coverage artifact
            uses: actions/upload-artifact@v2
            with:
              name: test-coverage-${{ matrix.python-version }}-${{ matrix.opt-extra }}
              path: coverage.xml

      # For analysis of codebase and submission to sonarcloud.io.
      sonarcloud:
        runs-on: ubuntu-latest

        # Requires coverage info generated from a previous unit-test run.
        needs: unittests

        steps:
          - uses: actions/checkout@v2
            with:
              fetch-depth: 0
          - name: Pull coverage XML from test run
            uses: actions/download-artifact@v2
            with:
              # Just one version of the coverage. Sonar can't merge a matrix of
              # coverages yet (2021-06).
              name: test-coverage-3.7-
          - name: Munge coverage.xml source path for SonarCloud container use
            # Bridge the gap between what pytest-cov outputs and SonarCloud repo
            # source directory assumptions of "/github/workspace/".
            # We check that the coverage source line is as expected before sed call.
            run: |
              grep -E '^\s*<source>.*</source>$' coverage.xml
              sed -Ei 's|^(\s*)<source>.*</source>|\1<source>/github/workspace/xaitk_saliency</source>|g' coverage.xml
          - name: SonarCloud Scan
            uses: sonarsource/sonarcloud-github-action@master
            env:
              GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
              SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}


This *requires* that a ``SONAR_TOKEN`` secret to be defined in the github
*repository* settings to be accessed within the workflow.
The application to the *repository* is important because PRs from forks will
not have access the secret defined in the upstream repository, thus the job
will fail for those fork-based PRs.

The value for this secret is from a SonarCloud personal security token (see
below on how to make one of these).
Currently, Paul Tunison holds the security token that is set to the
``SONAR_TOKEN`` secret in the upstream `xaitk-saliency`_ repository on GitHub.
In the future this may be changed by a repo admin, as described in a below
section.

Create a Personal Security Token
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Go to `SonarCloud`_.

* At the drop-down user option in the upper right --> select "My Account".

* Click "Security" tab.

* Enter the descriptive label for the token in the editable box --> click "Generate".

* Retain one-time-exposed value of token appropriately.

Set GitHub Repository ``SONAR_TOKEN`` Secret
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Go to the `XAITK-Saliency`_ repository page.

* Click on "Settings" --> "Secrets"

* If no existing ``SONAR_TOKEN`` secret, click on the "New repository secret"
  in the upper right.

  * This will open a new page to enter the name of the secret, which should be
    "SONAR_TOKEN" and a space to paste the value of the secret, which should be
    the token hash as generated above in `Creating a personal security token`_.

* Otherwise, update the existing secret value by clicking on the "Update"
  button to the right of the secret entry.

  * This will open a new page to enter a new value for the existing
    ``SONAR_TOKEN`` secret (i.e. cannot change the name of the secret).
    There should be a space to paste the value of the secret, which should be
    the token hash as generated above in `Creating a personal security token`_.


.. _SonarCloud: https://sonarcloud.io
.. _SonarCloud GitHub Action: https://github.com/SonarSource/sonarcloud-github-action
.. _XAITK-Saliency: https://github.com/XAITK/xaitk-saliency
