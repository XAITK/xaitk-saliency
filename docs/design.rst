Design Decisions
================

Concrete Dependencies and Updating
----------------------------------
This package, like many others, has dependencies that need to be fulfilled for
proper functionality to be satisfied.
While these general requirements, found in the :file:`pyproject.toml` file, are
known as "abstract" requirements, we also choose to codify "concrete"
requirements here via the :file:`poetry.lock` file.
This distinction is described well by [Stufft]_, using the ``setuptools``-based
resources which are parallel in use to the above respectively described files.

While xaitk-saliency is a library, we choose to retain concrete
dependencies via the :file:`poetry.lock` file to maintain consistency
of environment across developers as well as CI processes.
This falls in conceptual line with the "Developing Reusable Things or How Not
to Repeat Yourself" section from [Stufft]_.
However, we are still a "library" and desire to make sure that we work with the
"latest" versions of our listed abstract dependencies (within some reasonable
time window).
Currently, such concrete "version bumps" happen in the form of periodic update
branches that update the :file:`poetry.lock` file via a ``poetry update`` call.
These updates are submitted as PRs to the upstream repository and allow the
standard suite of CI checks to be performed to make sure the updated versions
do not break anything.
The timing of such updates are currently not concretely scheduled, nor are they
specifically tied to events, but more on an "every so often" cadence that is
relatively more frequent than versioned releases.

Image Format
------------
We choose to use the ``numpy.ndarray`` data structure for our image
representation in this toolkit.
Earlier, we utilized the Pillow package's ``PIL.Image.Image`` data structure
but encountered issues in certain use cases regarding large images, images with
non-standard quantities of channels (e.g. > 3) or with imagery consisting of
12 or 16-bit valuation.
Additionally, other popular and highly utilized packages in the Python
community, like OpenCV, Scikit-Image, and PyTorch to name a few, utilize raw
``numpy.ndarray`` matrices as the container for image data.


.. [Stufft] https://caremad.io/posts/2013/07/setup-vs-requirement/
