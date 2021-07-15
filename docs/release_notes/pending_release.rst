Pending Release Notes
=====================


Updates / New Features
----------------------

Documentation

* Add "SuperPixelSaliency" notebook to demonstrate use of arbitrary perturbation
  masks based on superpixels for saliency map generation.

Interfaces

* Update ``PerturbImage`` to only output perturbation masks, dropping physical
  image perturbation output. Since this output generation was the same across
  all known implementations, that part has been split out into a utility
  function.

* Added support for positive and negative saliency values as output by the
  saliency map generation interfaces.

* Updated ``PerturbImage`` interface to take in `numpy.ndarray` as the image
  data structure.

Implementations

* Add ``RISEScoring`` implementation, with the ability to also compute a
  de-biased form of RISE with an optional input parameter.

Utils

* Masking

  * Added utility functions for occluded image generation that was previously
    duplicated across ``PerturbImage`` implementations. Added both batch and
    streaming versions of this utility.


Fixes
-----

Documentation

* Fixed misspelled "miscellaneous" file.

Implementations

* Fix saliency map normalization in both ``OcclusionScoring`` as well as
  ``SimilarityScoring`` to disallow cross-class pollution in the norm.
