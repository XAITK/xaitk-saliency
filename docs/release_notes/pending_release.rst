XAITK-Saliency Pending Release Notes
====================================


Updates / New Features
----------------------

CI

* Added properties file for SonarQube scans.

Documentation

* Updated the "Occlusion Saliency" notebook to flow smoother and include
  un-commentable RISE perturbation algorithm option. The narrative has
  been more explicitly tuned to follow an "application" narrative.

* Add miscellaneous documentation on how to run a local SonarQube scan and
  experimental documentation on setting up scanning as a CI workflow job.

Interfaces

* Add new interfaces in accordance to the v0.1 API draft.

  * Added to doc-strings to expand on detail around saliency heatmap return
    value range and meaning.

  * Updated image perturbation interface to function in a streaming iterator
    fashion instead of in-bulk as a means of performance optimization as well
    as to allow it to function on larger image sizes and larger perturbation
    quantities at the same time.

Implementations

* Add new occlusion based classifier scoring in accordance to the v0.1 API draft for ImageClassifierSaliencyMapGenerator.

* Add new RISE based perturbation algorithm in accordance to the v0.1 API draft for PerturbImage

* Add new similarity based scoring algorithm in accordance to the v0.1 API draft for ImageSimilaritySaliencyMapGenerator

* Remove old "stub" implementations in transitioning to the new API draft

  * Removed "ImageSaliencyMapGenerator" interface class.

  * Removed "LogitImageSaliencyAugmenter" implementation class.

  * Removed "LogitImageSaliencyMapGenerator" implementation class.

  * Removed old RISE implementation classes.


Fixes
-----

* Update Read the Docs documentation link in README
