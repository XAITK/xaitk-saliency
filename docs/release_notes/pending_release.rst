XAITK-Saliency Pending Release Notes
====================================


Updates / New Features
----------------------

Documentation

* Updated the "Occlusion Saliency" notebook to flow smoother and include
  un-commentable RISE perturbation algorithm option. The narrative has
  been more explicitly tuned to follow an "application" narrative.

Interfaces

* Add new interfaces in accordance to the v0.1 API draft.

  * Added to doc-strings to expand on detail around saliency heatmap return
    value range and meaning.


Implementations

* Add new occlusion based classifier scoring in accordance to the v0.1 API draft for ImageClassifierSaliencyMapGenerator.

* Add new RISE based perturbation algorithm in accordance to the v0.1 API draft for PerturbImage

* Remove old "stub" implementations in transitioning to the new API draft

  * Removed "ImageSaliencyMapGenerator" interface class.

  * Removed "LogitImageSaliencyAugmenter" implementation class.

  * Removed "LogitImageSaliencyMapGenerator" implementation class.

  * Removed old RISE implementation classes.


Fixes
-----

* Update Read the Docs documentation link in README
