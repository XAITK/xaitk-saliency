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

Interfaces

* Added new high-level interface for black-box object detector saliency,
  ``GenerateObjectDetectorBlackboxSaliency``.

Implementations

* Added three ``GenerateObjectDetectorBlackboxSaliency`` implementations: the
  generic ``PerturbationOcclusion``, and two usable classes ``DRISEStack``
  and ``RandomGridStack``.

Examples

* Updated demo resource download links from Google Drive to data.kitware.com

Fixes
-----
