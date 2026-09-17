# Maintainer Workflow: Synchronizing GitLab and GitHub Repositories

This guide is intended for maintainers of both the GitLab and GitHub
repositories. The GitHub repository (`<remote>`) serves as the **primary**
codebase, while the GitLab repository (`origin`) is the **secondary**
repository. However, due to JATIC program requirements, primary development must
occur on GitLab. This process provides a workaround to ensure that GitHub
(`<remote>`) remains the primary codebase while complying with development
requirements on GitLab.

## Steps

01. **Clone the Repository** Clone the repository from GitLab (this will be your
    `origin` remote):

    ```bash
    git clone git@gitlab.jatic.net:jatic/kitware/xaitk-saliency.git
    ```

    > This step only needs to be done once.

02. **Add Upstream Remote Repository** Add the GitHub repository as an upstream
    remote repository named `<remote>`:

    ```bash
    git remote add <remote> git@github.com:XAITK/xaitk-saliency.git
    ```

    > This step only needs to be done once.

03. **Sync Local Branch** Ensure your local branch `main` is up-to-date with
    `origin/main`:

    ```bash
    git fetch origin
    git checkout main
    git pull origin main
    ```

04. **Push to <Remote> Remote Repository** Push your `main` branch to the
    `<remote>` remote:

    ```bash
    git push <remote> main
    ```

05. **Create a Pull Request** Open a pull request from `main` to the target
    branch on the `<remote>` remote repository.

06. **Follow Contribution Guidelines** Refer to
    [CONTRIBUTING.md](CONTRIBUTING.md) for details on pull request code reviews
    and merging.

07. **Mirror Updates to Origin** Ensure `origin/master` mirrors
    `<remote>/master`.

    > This process happens automatically but may take a few minutes to reflect
    > changes.

08. **Update Local Master Branch** Sync your local `master` branch with
    `origin/master`:

    ```bash
    git fetch origin
    git checkout master
    git pull origin master
    ```

09. **Switch to Main** Check out your `main` branch:

    ```bash
    git checkout main
    ```

10. **Rebase Master onto Main** Rebase `<remote>` onto `main`:

    ```bash
    git rebase master
    ```

11. **Push Main to Origin** Push the updated `main` branch back to the `origin`
    remote:

    ```bash
    git push origin main
    ```
