# XAITK-Saliency Examples
This directory hosts the XAITK-Saliency examples.

### Requirements
Most of the examples require [Jupyter Notebook](https://jupyter.org/) and [Pytorch](https://pytorch.org/).

These can be installed manually using `pip` or with the following command:

```bash
poetry install -E example_deps
```

Some notebooks may require additional dependencies. Please see the first cell of each notebook ("Setup environment") on how to install the relevant packages.

### Run the notebooks from Colab

Most of the notebooks have an "Open in Colab" button.
Please right-click on the button, and select "Open Link in New Tab" to start a Colab page with the corresponding notebook content.

To use GPU resources through Colab, please remember to change the runtime type to `GPU`:

1. From the `Runtime` menu select `Change runtime type`
1. Choose `GPU` from the drop-down menu
1. Click `SAVE`
This will reset the notebook and may ask you if you are a robot (these instructions assume you are not).

Running:

```bash
!nvidia-smi
```

in a cell will verify this has worked and show you what kind of hardware you have access to.

**Please note that after setting up the environment, you may need to "Restart Runtime" in order to resolve package version conflicts.**

### Data

Some notebooks may require additional data. This data will be downloaded when running the notebook.

### Encountering Issues

For issues relating to XAITK-Saliency functionality or running of an example, please create an issue on the [repository](https://github.com/XAITK/xaitk-saliency/issues).


### Continuous Integration
Notebooks contained here are (mostly) executed as part of the continuous
integration workflows to ensure their example appropriately reflects the
toolkit's current functionality.
The CI workflow may be found in ``.github/workflows/ci-example-notebooks.yml``.

Not all are included in this CI due to their long-running or heavy resource
requirements.
The specification of which notebooks are included in CI is around L29 of the
above referenced workflow configuration.

The following is a list of rationales as to why certain notebooks are not
currently included in to CI workflow:

* ``covid_classification.ipynb`` which trains a deep learning model. This both
  takes a non-trivial amount of time and consumes greater resources.

---

This README is adapted from [MONAI Tutorials](https://github.com/Project-MONAI/tutorials).
