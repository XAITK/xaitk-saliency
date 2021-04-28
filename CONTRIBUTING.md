# Contributing to XAITK-Saliency

Here we describe at a high level how to contribute to XAITK-Saliency.
See the [XAITK-Saliency README] file for additional information.


## The General Process

1.  The official XAITK-Saliency source is maintained on Github.

2.  Fork XAITK-Saliency into your user's namespace and clone this repository
    on your system.

3.  Create a topic branch, edit files and create commits:

        $ git checkout -b <branch-name>
        $ <edit things>
        $ git add <file1> <file2> ...
        $ git commit

4.  Push topic branch with commits to your fork in GitHub:

        $ git push origin HEAD -u

5.  Visit the Kitware XAITK-Saliency Github, browse to the "Pull requests" tab
    and click on the "New pull request" button in the upper-right.
    Click on the "compare across forks" link, browse to your fork and browse to
    the topic branch to submit for the pull request.
    Finally, click the "Create pull request" button to create the request.


XAITK-Saliency uses GitHub for code review and Github Actions for continuous
testing as new pull requests are made.
All checks/tests must pass before a PR can be merged.

Sphinx is used for manual and automatic API [documentation].


[XAITK-Saliency README]: README.md
[documentation]: docs/
