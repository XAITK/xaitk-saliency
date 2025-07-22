![xaitk-logo](./docs/figures/xaitk-wordmark-light.png)

<hr/>

<!-- :auto badges: -->

[![PyPI - Python Version](https://img.shields.io/pypi/v/xaitk-saliency)](https://pypi.org/project/xaitk-saliency/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xaitk-saliency)
[![Documentation Status](https://readthedocs.org/projects/xaitk-saliency/badge/?version=latest)](https://xaitk-saliency.readthedocs.io/en/latest/?badge=latest)

<!-- :auto badges: -->
<!-- TODO: re-enable these later. -->
<!-- [![badge-unittests](https://github.com/xaitk/xaitk-saliency/actions/workflows/ci-unittests.yml/badge.svg)](https://github.com/XAITK/xaitk-saliency/actions/workflows/ci-unittests.yml)
[![badge-notebooks](https://github.com/xaitk/xaitk-saliency/actions/workflows/ci-example-notebooks.yml/badge.svg)](https://github.com/XAITK/xaitk-saliency/actions/workflows/ci-example-notebooks.yml) -->
[![codecov](https://codecov.io/gh/XAITK/xaitk-saliency/branch/master/graph/badge.svg?token=VHRNXYCNCG)](https://codecov.io/gh/XAITK/xaitk-saliency)

# XAITK - Saliency
The `xaitk-saliency` package is an open source, Explainable AI (XAI) framework
for visual saliency algorithm interfaces and implementations, built for
analytics and autonomy applications.

See [here](https://xaitk-saliency.readthedocs.io/en/latest/xaitk_explanation.html)
for a more formal introduction to the topic of XAI and visual saliency
explanations.

This framework is a part of the [Explainable AI Toolkit (XAITK)](
https://xaitk.org).

## Supported Algorithms
The `xaitk-saliency` package provides saliency algorithms for a wide range of image understanding
tasks, including image classification, image similarity, object detection, and reinforcement learning.
The current list of supported saliency algorithms can be found [here](
https://xaitk-saliency.readthedocs.io/en/latest/introduction.html#saliency-algorithms).

## Target Audience
This toolkit is intended to help data scientists and developers who want to
add visual saliency explanations to their workflow or product.
Functionality provided here is both directly accessible for targeted
experimentation, and through [Strategy](
https://en.wikipedia.org/wiki/Strategy_pattern) and [Adapter](
https://en.wikipedia.org/wiki/Adapter_pattern) patterns to allow for
modular integration into systems and applications.

<!-- :auto installation: -->

## Installation

`xaitk-saliency` installation has been tested on Unix and Linux systems.

To install the current version via `pip`:

```bash
pip install xaitk-saliency[<extra1>,<extra2>,...]
```

To install the current version via `conda-forge`:

```bash
conda install -c conda-forge xaitk-saliency
```

Certain plugins may require additional runtime dependencies. Details on these
requirements can be found
[here](https://xaitk-saliency.readthedocs.io/en/latest/implementations.html).

For more detailed installation instructions, visit the
[installation documentation](https://xaitk-saliency.readthedocs.io/en/latest/installation.html).

<!-- :auto installation: -->

<!-- :auto getting-started: -->

## Getting Started

Explore usage examples of the `xaitk-saliency` package in various contexts using
the Jupyter notebooks provided in the `./docs/examples/` directory.

<!-- :auto getting-started: -->

<!-- :auto documentation: -->

## Documentation

Documentation for both release snapshots and the latest main branch is available
on [ReadTheDocs](https://xaitk-saliency.readthedocs.io/en/latest/).

To build the Sphinx-based documentation locally for the latest reference:

```bash
# Install dependencies
poetry install --sync --with main,linting,tests,docs
# Navigate to the documentation root
cd docs
# Build the documentation
poetry run make html
# Open the generated documentation in your browser
firefox _build/html/index.html
```

<!-- :auto documentation: -->

<!-- :auto contributing: -->

## Contributing

Contributions are encouraged!

The following points help ensure contributions follow development practices.

- Follow the
  [JATIC Design Principles](https://cdao.pages.jatic.net/public/program/design-principles/).
- Adopt the Git Flow branching strategy.
- Detailed release information is available in
  [docs/release_process.rst](./docs/release_process.rst).
- Additional contribution guidelines and issue reporting steps can be found in
  [CONTRIBUTING.md](./CONTRIBUTING.md).

<!-- :auto contributing: -->

<!-- :auto developer-tools: -->

### Developer Tools

Ensure the source tree is acquired locally before proceeding.

#### Poetry Install

You can install using [Poetry](https://python-poetry.org/):

> [!IMPORTANT] XAITK-Saliency currently requires `poetry<2.0`

> [!WARNING] Users unfamiliar with Poetry should use caution. See
> [installation documentation](https://xaitk-saliency.readthedocs.io/en/latest/installation.html#from-source)
> for more information.

```bash
poetry install --with main,linting,tests,docs --extras "<extra1> <extra2> ..."
```

#### Pre-commit Hooks

Pre-commit hooks ensure that code complies with required linting and formatting
guidelines. These hooks run automatically before commits but can also be
executed manually. To bypass checks during a commit, use the `--no-verify` flag.

To install and use pre-commit hooks:

```bash
# Install required dependencies
poetry install --sync --with main,linting,tests,docs
# Initialize pre-commit hooks for the repository
poetry run pre-commit install
# Run pre-commit checks on all files
poetry run pre-commit run --all-files
```

<!-- :auto developer-tools: -->

## Example: A First Look at xaitk-saliency
This [associated project](https://github.com/XAITK/xaitk-saliency-web-demo)
provides a local web-application that provides a demonstration of visual
saliency generation in a user-interface.
This provides an example of how visual saliency, as generated by this package,
can be utilized in a user-interface to facilitate model and results
exploration.
This tool uses the [trame framework](https://kitware.github.io/trame/).

| ![image1](https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/main/gallery/xaitk-classification-rise-4.jpg) | ![image2](https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/main/gallery/xaitk-classification-sliding-window.jpg) | ![image3](https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/main/gallery/xaitk-detection-retina.jpg) | ![image4](https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/main/gallery/xaitk-similarity-1.jpg) |
|:---------:|:---------:|:---------:|:---------:|


<!-- :auto license: -->

## License

[Apache 2.0](./LICENSE)

<!-- :auto license: -->

All development prior to Nov 19, 2024 falls under [BSD-3-Clause](./LICENSE.old)

<!-- :auto contacts: -->

## Contacts

**Principal Investigator / Product Owner**: Brian Hu (Kitware) @brian.hu

**Scrum Master / Maintainer**: Brandon RichardWebster (Kitware)
@b.richardwebster

**Deputy Scrum Master / Maintainer**: Emily Veenhuis (Kitware) @emily.veenhuis

**Project Manager**: Keith Fieldhouse (Kitware) @keith.fieldhouse

**Program Representative**: Austin Whitesell (MITRE) @awhitesell

<!-- :auto contacts: -->

<!-- :auto acknowledgment: -->

## Acknowledgment

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. 519TC-23-9-2032. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.

<!-- :auto acknowledgment: -->
