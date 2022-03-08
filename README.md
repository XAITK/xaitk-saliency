![xaitk-logo](./docs/figures/xaitk-wordmark.png)

<hr/>

[![PyPI - Python Version](https://img.shields.io/pypi/v/xaitk-saliency)](https://pypi.org/project/xaitk-saliency/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xaitk-saliency)
[![Documentation Status](https://readthedocs.org/projects/xaitk-saliency/badge/?version=latest)](https://xaitk-saliency.readthedocs.io/en/latest/?badge=latest)
[![badge-unittests](https://github.com/xaitk/xaitk-saliency/actions/workflows/ci-unittests.yml/badge.svg)](https://github.com/XAITK/xaitk-saliency/actions/workflows/ci-unittests.yml)
[![badge-notebooks](https://github.com/xaitk/xaitk-saliency/actions/workflows/ci-example-notebooks.yml/badge.svg)](https://github.com/XAITK/xaitk-saliency/actions/workflows/ci-example-notebooks.yml)
[![codecov](https://codecov.io/gh/XAITK/xaitk-saliency/branch/master/graph/badge.svg?token=VHRNXYCNCG)](https://codecov.io/gh/XAITK/xaitk-saliency)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/XAITK/xaitk-saliency.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/XAITK/xaitk-saliency/context:python)

# XAITK - Saliency
The `xaitk-saliency` package is an open source, Explainable AI (XAI) framework
for visual saliency algorithm interfaces and implementations, built for
analytics and autonomy applications.

See [here](https://xaitk-saliency.readthedocs.io/en/latest/introduction.html)
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

## Installation
Install the latest release via pip:
```bash
pip install xaitk-saliency
```

Some plugins may require additional dependencies in order to be utilized at
runtime.
Such details are described [here](
https://xaitk-saliency.readthedocs.io/en/latest/implementations.html).

See [here for more installation documentation](
https://xaitk-saliency.readthedocs.io/en/latest/installation.html).

## Getting Started
We provide a number of examples based on Jupyter notebooks in the `./examples/`
directory to show usage of the `xaitk-saliency` package in a number of
different contexts.

Contributions are welcome!
See the [CONTRIBUTING.md](./CONTRIBUTING.md) file for details.

## Documentation
Documentation snapshots for releases as well as the latest master are hosted on
[ReadTheDocs](https://xaitk-saliency.readthedocs.io/en/latest/).

The sphinx-based documentation may also be built locally for the most
up-to-date reference:
```bash
# Install dependencies
poetry install
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```
