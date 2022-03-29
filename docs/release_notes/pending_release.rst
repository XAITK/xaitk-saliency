Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Added the ATARI example notebook to the list of notebooks to run during CI.

Documentation

* Update saliency algorithms table with perturbation-based saliency for reinforcement learning
  and add corresponding section to README.

* Added a lighter color version of the logo that will appear better is both
  light- and dark-theme contexts. The main README file has been updated to refer
  to this image.

Examples

* Updated demo resource download links from Google Drive to data.kitware.com

* Added example using saliency to qualitatively compare two object detection
  models.

Interfaces

* Added new high-level interface for black-box object detector saliency,
  ``GenerateObjectDetectorBlackboxSaliency``.

* Added new high-level interface for image similarity saliency,
  ``GenerateImageSimilarityBlackboxSaliency``.

Implementations

* Added three ``GenerateObjectDetectorBlackboxSaliency`` implementations: the
  generic ``PerturbationOcclusion``, and two usable classes ``DRISEStack``
  and ``RandomGridStack``.

* Updated behavior of the ``SlidingWindow`` ``PerturbImage`` implementation. For
  a given stride, the number of masks generated is now agnostic to the window
  size.

Misc.

* Updated `poetry-core` build backend to version `1.0.8`, which now supports
  `pip` editable installs (`pip install -e .`).

Utils

* Updated COCO utility functions to use new high-level detector interface.
  `gen_coco_sal()` is now deprecated in exchange for `parse_coco_dset()` which
  parses a `kwcoco.CocoDataset` object into the inputs used with an
  implementation of `GenerateObjectDetectorBlackboxSaliency`.

Fixes
-----
