Local SonarQube Testing
=======================
Follow the `Try Out SonarQube`_ documentation with the "From the Docker
image" method.
As part of the setup of the repository/project you will need to create a token
for running scans with.
The below assumes that you have saved this locally somewhere for reference,
like the example file `~/sonarqube-local-token`.

From the above you will learn that you will need to run the ``sonar-scanner``
separately from running the local server.
Documentation and references to acquire the scanner maybe found on their
`SonarScanner`_ docs page.

Locally, we have had success running a script as per the following:

.. code-block:: bash

    #! /bin/bash
    SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    docker run \
      --rm \
      --network host \
      -e SONAR_HOST_URL="http://localhost:9000" \
      -e SONAR_LOGIN="$(cat "$HOME"/sonarqube-local-token)" \
      -v "${SOURCE_DIR}:${SOURCE_DIR}" \
      -w "${SOURCE_DIR}" \
      sonarsource/sonar-scanner-cli



.. _Try Out SonarQube: https://docs.sonarqube.org/latest/setup/get-started-2-minutes/
.. _SonarScanner: https://docs.sonarqube.org/latest/analysis/scan/sonarscanner/
